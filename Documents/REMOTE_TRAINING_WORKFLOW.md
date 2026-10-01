# Remote training workflow (SSH / VSCode)

How to train on this machine while working from another machine over SSH
(e.g. VSCode Remote-SSH), and how to watch the run live in MLflow from there.

## Why this works

- `mlflow.tracking_uri` in every config is `http://127.0.0.1:5000` (loopback
  only) - the MLflow server and the training process both run on this
  machine, so nothing needs to be exposed to the network.
- This machine does **not** kill background processes on SSH logout
  (`KillUserProcesses=no`), so a `tmux` session keeps running after you
  disconnect - no `systemd`/`sudo` setup needed.
- `start_mlflow.sh` and `run_training.sh` (repo root) are the Linux
  equivalents of `start_mlflow.bat`/`run_training.bat`, adapted to launch
  into a named tmux session instead of a new console window.
- They use a **dedicated tmux server, `tmux -L fiber`**, started without
  D-Bus. The machine runs `systemd-oomd`, which kills processes inside
  `user@1007.service` when memory pressure spikes. On this 2 TB box that
  pressure comes from reclaiming ~1.8 TB of page cache during heavy IO, not
  from a real memory shortage. A plain `tmux` puts every pane in that monitored
  cgroup, and on 2026-09-25 oomd took down the desktop session, the tmux server
  and MLflow. Panes of the `fiber` server stay in your SSH/VS Code session
  scope, which oomd does not watch. **Always add `-L fiber`** to tmux commands
  for these jobs; a plain `tmux ls` won't show them.

## One-time setup (already done)

- `conda env create -f environment.yaml` -> `cnn_test` env (verify with
  `conda env list`; re-run `pytest` after any environment change).
- MLflow server started once in a persistent tmux session named `mlflow`.
  It stays up indefinitely - you should not need to start it again unless
  the machine reboots.

## Starting a training run

```bash
cd "/work/cfp406/NFA SEM Asbestos/fiber unet trainer"
./run_training.sh configs/example.yaml                  # session name "train"
./run_training.sh -n train_b configs/a.yaml configs/b.yaml   # a queue, in a 2nd session
```

This validates every config first, starts MLflow if it isn't already running, then
runs `python -m fiberseg.train --config <path>` for each config in turn inside its
own tmux session (a failed config is logged and the queue continues). Output is
mirrored to `logs/<session>_<timestamp>.log`. Closing your SSH/VSCode connection
does **not** stop it. A finished session stays open so you can read the end of
the output; remove it with `tmux -L fiber kill-session -t <name>` before reusing
the name.

Useful commands:

```bash
tmux -L fiber ls                                # list sessions (train, mlflow, ...)
tmux -L fiber attach -t train                   # watch live; detach with Ctrl-b then d
tmux -L fiber capture-pane -pt train | tail -40 # peek without attaching
tmux -L fiber kill-session -t train             # stop a run
tail -f logs/train_*.log                        # or just follow the log file
```

If a run ever disappears without an error in its log, check
`journalctl --user --since today | grep -i oomd`.

## Watching MLflow from the other machine

**Option A - VSCode Remote-SSH (recommended, no setup):** once connected to
this machine in VSCode, MLflow listening on port 5000 is usually detected
automatically and forwarded. Open the **Ports** tab in the bottom panel; if
`5000` isn't listed, click **Forward a Port** and enter `5000`. Then
ctrl/cmd-click the forwarded address (or the globe icon) to open the MLflow
UI in your local browser.

**Option B - manual SSH tunnel (works from any terminal, VSCode or not):**

```bash
ssh -L 5000:localhost:5000 cfp406@<this-machine-hostname-or-ip>
```

Then open `http://localhost:5000` in a browser on the other machine. To
avoid retyping this, add to `~/.ssh/config` on the other machine:

```
Host fiber-trainer
    HostName <this-machine-hostname-or-ip>
    User cfp406
    LocalForward 5000 localhost:5000
```

then just `ssh fiber-trainer` and browse `localhost:5000`.

## Asking Claude to start training

With the above in place, once connected to this machine (via VSCode
Remote-SSH or plain SSH) you can just ask Claude Code to start a run - it
will call `./run_training.sh <config>` the same way described above, check
`tmux -L fiber capture-pane` to confirm it started cleanly, and report back the
session name to attach to or the MLflow run to watch.

## After a run: tuned thresholds

Every run tunes its decision threshold(s) on the validation split after
training and stores them in the best checkpoint, so `predict_tiles` /
`predict_all` / `export_model.py` use them automatically. For the per-mode test
results, see the run's MLflow metrics `tuned_test/<mode>/*` or
`tuned_thresholds.json` in its checkpoint folder. To compare the arms of a
multi-arm study:

```bash
python -m fiberseg.tools.summarize_experiment --experiment proxy-r34-scratch-ablation
```
