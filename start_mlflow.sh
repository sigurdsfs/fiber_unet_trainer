#!/usr/bin/env bash
# Linux equivalent of start_mlflow.bat: starts the MLflow tracking UI in a
# persistent tmux session named "mlflow" on the dedicated "fiber" tmux server,
# so it survives SSH disconnects. Safe to run repeatedly: no-ops if MLflow is up.
#
# Why a dedicated server started without D-Bus: this machine runs systemd-oomd,
# which kills processes inside user@<uid>.service when memory pressure spikes
# (on a 2 TB box that pressure comes from page-cache reclaim during heavy IO, not
# from real memory shortage). tmux 3.4 normally moves each pane into a
# tmux-spawn-*.scope under user@<uid>.service - i.e. into oomd's reach - and that
# killed a queue on 2026-09-25. Hiding the user bus makes panes stay in the
# launching SSH/VS Code session scope, which oomd does not monitor.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SESSION="mlflow"
URL="http://127.0.0.1:5000"
TMUX_CMD=(env -u DBUS_SESSION_BUS_ADDRESS -u XDG_RUNTIME_DIR tmux -L fiber)

if curl -s -o /dev/null -m 1 "$URL"; then
    echo "MLflow is already running at $URL"
    exit 0
fi

if "${TMUX_CMD[@]}" has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists but MLflow isn't responding yet."
    echo "Check it with: tmux -L fiber attach -t $SESSION"
    exit 0
fi

CONDA_BASE="$(conda info --base)"
"${TMUX_CMD[@]}" new-session -d -s "$SESSION" \
    "source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate cnn_test && MLFLOW_ALLOW_FILE_STORE=true python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000 --workers 1"

echo "Starting MLflow in tmux session '$SESSION' (attach with: tmux -L fiber attach -t $SESSION)"
echo "Waiting for it to come up..."
for _ in $(seq 1 60); do
    if curl -s -o /dev/null -m 1 "$URL"; then
        echo "MLflow is up at $URL"
        exit 0
    fi
    sleep 1
done
echo "WARNING: MLflow did not respond within 60s - check: tmux -L fiber attach -t $SESSION"
