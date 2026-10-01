#!/usr/bin/env bash
# Linux equivalent of run_training.bat: ensures MLflow is up, then launches
# training in its own persistent tmux session so it keeps running after you
# disconnect (SSH/VSCode Remote-SSH session ending does not kill it).
#
# Usage: ./run_training.sh [-n session-name] <config> [config ...]
#   Several configs run one after another in the same session; a failed config
#   is logged and the queue moves on. session-name defaults to "train"; start a
#   second session with a different name to run two queues on the GPU at once.
#
# Runs on the dedicated "fiber" tmux server (tmux -L fiber ...) started without
# D-Bus, so panes stay out of systemd-oomd's reach - see start_mlflow.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TMUX_CMD=(env -u DBUS_SESSION_BUS_ADDRESS -u XDG_RUNTIME_DIR tmux -L fiber)

usage() { echo "Usage: $0 [-n session-name] <config> [config ...]" >&2; exit 2; }
SESSION="train"
while getopts "n:h" opt; do
    case "$opt" in
        n) SESSION="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[ "$#" -ge 1 ] || usage

if "${TMUX_CMD[@]}" has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists - a training run may already be active."
    echo "Attach with: tmux -L fiber attach -t $SESSION   (or remove it: tmux -L fiber kill-session -t $SESSION)"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cnn_test

# Fail now, not hours into the queue, on a missing or invalid config.
for cfg in "$@"; do
    python -c "import sys; from fiberseg.config import load_config; load_config(sys.argv[1])" "$cfg" \
        || { echo "Invalid config: $cfg" >&2; exit 1; }
done

./start_mlflow.sh

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/${SESSION}_${STAMP}.log"
QUEUE="logs/${SESSION}_${STAMP}.queue.sh"
{
    echo "#!/usr/bin/env bash"
    echo "source $(printf %q "$CONDA_BASE/etc/profile.d/conda.sh") && conda activate cnn_test"
    echo "cd $(printf %q "$PWD")"
    printf 'for cfg in'; printf ' %q' "$@"; echo '; do'
    echo '    echo "=== $(date "+%F %T") starting $cfg"'
    echo '    python -m fiberseg.train --config "$cfg" \'
    echo '        || echo "!!! $(date "+%F %T") FAILED: $cfg (continuing with the next config)"'
    echo 'done'
    echo 'echo "=== $(date "+%F %T") queue finished"'
} > "$QUEUE"
chmod +x "$QUEUE"

# remain-on-exit keeps the finished pane (so status/errors stay readable);
# pipe-pane mirrors all output to a log file without changing tty behaviour.
"${TMUX_CMD[@]}" new-session -d -s "$SESSION" "bash $(printf %q "$QUEUE")" \; \
    set-option -t "$SESSION" remain-on-exit on \; \
    pipe-pane -t "$SESSION" "cat >> $(printf %q "$LOG")"

echo "Started tmux session '$SESSION' with $# config(s):"
printf '  %s\n' "$@"
echo "Log:     $LOG"
echo "Attach:  tmux -L fiber attach -t $SESSION   (detach again with Ctrl-b then d)"
echo "Status:  tmux -L fiber capture-pane -pt $SESSION | tail -40"
echo "Stop:    tmux -L fiber kill-session -t $SESSION"
