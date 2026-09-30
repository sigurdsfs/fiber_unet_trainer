"""Step 1 - DLL loads, server name is right, login works, disconnect leaves ESPRIT running.

Answers: DLL path/bitness, exact server name (e.g. "Lokaler Server"), credentials.
"""

import ctypes

from .. import bruker_api as ba
from ._common import connected, parse


def main():
    _, cfg = parse(__doc__)
    api = ba.BrukerAPI(cfg.dll_path)
    print(f"loaded {cfg.dll_path}; all {len(ba.SIGNATURES)} bound functions found")

    buf = ctypes.create_string_buffer(4096)
    api.call("QueryServers", buf, len(buf))
    print("servers:", buf.value.decode("cp1252", errors="replace"))  # TODO: separator format?

    with connected(cfg) as scope:
        print("info:", scope.query_info())


if __name__ == "__main__":
    main()
