"""Step 6 - one small stage move and back, with image-based check of direction and scale.

    python -m fiberseg.sem_eds.bringup.step6_stage --dx-um 5

Answers: stage units (mm), which GetSEMStageState value = idle, and the sign/rotation of the
stage axes relative to the image axes (-> settings.stage_x_sign / stage_y_sign).
Requires stage_x/y_limits_um to be set in the local settings.
"""

import numpy as np
from skimage.registration import phase_cross_correlation

from ._common import confirm, connected, parse


def main():
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--dx-um", type=float, default=5.0))
    with connected(cfg) as scope:
        scope.setup(cfg)
        x0, y0, *_ = scope.stage_position_mm()
        x0_um, y0_um = x0 * 1000, y0 * 1000
        target = (x0_um + args.dx_um, y0_um)
        scope.check_stage_target(*target)  # raises before anything moves
        confirm(f"Move stage x {x0_um:.1f} -> {target[0]:.1f} um, then back.")

        before = scope.acquire_image()
        print("stage state before:", scope.stage_state())
        scope.move_stage(*target)
        print("stage state after:", scope.stage_state())
        after = scope.acquire_image()
        scope.move_stage(x0_um, y0_um)

    (dy, dx), _, _ = phase_cross_correlation(before, after, upsample_factor=10)
    moved_um = np.hypot(dx, dy) * cfg.pixel_size_um
    angle = np.degrees(np.arctan2(dy, dx))
    print(f"image shift: dx={dx:.1f} px, dy={dy:.1f} px = {moved_um:.2f} um, angle {angle:.1f} deg")
    print(f"commanded: {args.dx_um} um along stage x")
    scale_ok = abs(moved_um - abs(args.dx_um)) / abs(args.dx_um) < 0.1
    print("-> scale ok" if scale_ok else "-> SCALE MISMATCH")
    print("Angle ~180 deg -> set stage_x_sign: -1.")
    print("Angle not near 0/90/180/270 deg -> scan rotation must be handled first.")


if __name__ == "__main__":
    main()
