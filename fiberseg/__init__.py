import os
import sys
import warnings

# On Windows, stdout/stderr default to the system codepage (e.g. cp1252), which
# can't encode emoji that dependencies print (mlflow's "View run" links, Lightning
# tips, etc.) - crashing with UnicodeEncodeError. This mainly bites when output is
# redirected to a file or pipe (an interactive console is often more lenient), e.g.
# `python -m fiberseg.train ... > log.txt` for an unattended sweep.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

# Must be set before `albumentations` is first imported anywhere (main process or
# any DataLoader worker process) to skip its online PyPI version check. Without
# this it fires a network call - and a warning when that call fails - once per
# process, since a warnings.filterwarnings() call in one module doesn't reach
# other processes that import albumentations independently (e.g. spawned
# DataLoader workers on Windows).
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Harmless Lightning/torch pytree API deprecation warning (Lightning-internal,
# not something this codebase controls); silenced for the same cross-process
# reason as above.
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=UserWarning,
)
