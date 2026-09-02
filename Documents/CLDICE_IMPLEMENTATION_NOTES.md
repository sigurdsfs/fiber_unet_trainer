# clDice implementation notes

`fiberseg/lit_module.py`'s `_soft_skeletonize`/`_soft_cldice` (the soft
centerline-Dice topology term added on top of the base loss when
`train.loss.cldice_weight > 0`) is a reimplementation of Shit et al.'s
soft-skeletonization clDice loss. It was checked line-by-line against the
authors' reference implementation:

- https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py
- https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py

Two real deviations were found and fixed (see git history around this file for
the change): erosion was using a full 3x3 square structuring element instead
of the reference's cross-shaped (4-connected) one - the square erosion is more
aggressive and was destroying thin *diagonal* fiber structure faster across
iterations than the reference algorithm intends - and the skeleton
accumulation step was missing a defensive `relu()` clamp the reference has.

Two further differences were found that are **not bugs** - both look like
deliberate, reasonable choices already consistent with the rest of this
file - but are recorded here as possible future changes if the clDice term's
behavior ever needs tuning or needs to match the reference paper exactly.

## 1. Per-sample vs. pooled-batch averaging

**Reference** (`cldice.py`): sums `tprec`/`tsens` over the *entire batch* in
one shot (plain `torch.sum(...)` with no per-sample dimension), so one
scalar clDice score is computed per batch, pooling every sample's skeleton
pixels together.

**Current** (`lit_module.py::_soft_cldice`): sums only over the spatial
dimensions (`dims = tuple(range(1, probs.ndim))`), producing one clDice score
*per sample*; the caller (`_loss`) then averages those per-sample scores
(`cldice.mean()`).

Practical difference: with pooled-batch averaging, a sample with a lot of
foreground (many skeleton pixels) dominates the batch's clDice score and
therefore the gradient contribution from thin/sparse samples shrinks
proportionally. With per-sample averaging (current), every sample contributes
equally to the loss regardless of how much foreground it has - arguably a
better fit for this project, where tile foreground fraction varies a lot
(see `data.min_foreground_fraction`/`keep_empty_probability`), but it is a
real behavioral difference from the published reference.

**If revisiting:** switching to pooled-batch averaging would mean summing
`skel_pred * target` etc. over `dim=None` (the whole batch) before dividing,
and `_loss()` would no longer need `.mean()` over the returned value (it
would already be a scalar). Would need re-tuning `train.loss.cldice_weight`
alongside, since the loss magnitude/gradient scale would change.

## 2. Smoothing constant magnitude

**Reference**: `smooth = 1.0` (default), added to both the numerator and
denominator of `tprec`/`tsens` - a fairly strong Laplace-style smoothing that
noticeably damps the ratio toward the smoothing prior when the skeleton has
only a few pixels (e.g. a tile with a very short fiber fragment).

**Current**: `eps = 1e-7`, the same tiny epsilon used everywhere else in this
file (`_soft_tversky_index`, `_stats_from_counts`) purely to avoid literal
division by zero, with no meaningful smoothing effect otherwise.

Both give the same answer in the fully-degenerate case (no skeleton pixels at
all: `(0+c)/(0+c) = 1` for any constant `c`), but they diverge for small,
nonzero skeleton pixel counts - `eps=1e-7` lets `tprec`/`tsens` swing close to
0 or 1 based on just a handful of pixels, while `smooth=1.0` pulls those same
cases much closer to a neutral value. `eps=1e-7` is consistent with this
codebase's existing convention, so it was left as-is rather than special-cased
just for clDice.

**If revisiting:** this is a single-constant change
(`eps: float = 1e-7` -> a larger value, or a separate `smooth` parameter) in
`_soft_cldice`. Worth an ablation if the cldice term looks noisy/unstable on
very sparse tiles specifically, since that's exactly the regime where this
constant matters.
