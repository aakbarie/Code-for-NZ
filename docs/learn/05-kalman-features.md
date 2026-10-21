# 05 · Features along flight lines with a Kalman filter

So far every sounding has been treated on its own: the inversion fits each one
separately, and the histogram model classifies each one separately. But the
helicopter samples every 3.2 m, and the ground doesn't change much in 3 m.
The raw channels are about 98% correlated 32 m apart and still 60–87% at
320 m. A state-space model makes that continuity explicit.

## The model

```
x_t = A x_{t-1} + w_t          hidden state: 4 numbers describing the ground under sounding t
y_t = C x_t + D u_t + v_t      what the bird sees: 12 channels (log ppm, standardised)
```

- **x_t, the state:** not resistivity at given depths, but whatever 4 numbers
  best explain the 12 channels.
- **A:** how the state carries over from one sounding to the next. Close to
  the identity matrix here, because the ground changes slowly.
- **w_t, process noise (covariance Q):** how much the ground is *allowed* to
  change per step. Small Q gives smooth features; large Q lets them jump.
- **C:** how the state shows up in each channel.
- **u_t, inputs:** things we know and don't want in the state. Here that's a
  constant, log bird height, and its square. **D** is their effect, learned
  from the data. This matters because log amplitude correlates with log
  height at up to −0.90 (40 kHz quadrature). Without this term, the "geology"
  features would partly be the pilot's altitude.
- **v_t, observation noise (variances R, one per channel):** sensor noise and
  anything the 4-number state can't explain.

## The three algorithms

**Kalman filter (forward).** At each sounding:
1. *Predict* the state from the previous one: x̂ = A x, with uncertainty
   P = A P A′ + Q.
2. Compare with what was measured. The **innovation**, e = y − C x̂ − D u, is
   the surprise.
3. *Update*: move the state toward the measurement by the Kalman gain
   K = P C′ S⁻¹, where S = C P C′ + R is the innovation's expected spread.
   The gain weighs how much to trust the new sounding against the running
   estimate.

The **innovation score** e′ S⁻¹ e should follow a χ² distribution with 12
degrees of freedom if the model is right. Large values mean "this sounding
doesn't fit its neighbours": a spike, power-line noise, or a sharp
geological boundary.

**Rauch–Tung–Striebel smoother (backward).** The filter only uses soundings
*before* t. The smoother runs back along the line, so each estimate uses the
whole line, before and after. Its uncertainty is never larger than the
filter's.

**EM (Shumway & Stoffer, 1982): learning A, C, D, Q, R.**
- **E-step:** run the filter and smoother with the current parameters.
- **M-step:** re-estimate every parameter in closed form from the smoothed
  states, as linear regressions.
- **Repeat.** Each round can only increase the likelihood; the tests check
  this. All 12 lines share one set of parameters, and each line is its own
  sequence.

The filter, smoother and EM are about 150 lines of numpy in
[`src/morrill/analysis/statespace.py`](../../src/morrill/analysis/statespace.py),
written to be read.

## One subtlety: the features have no fixed "direction"

C x = (C T⁻¹)(T x) for any invertible T, so the model can't tell one set of
4 numbers from a rotated set. After fitting, the code picks a canonical
rotation: the features are uncorrelated with unit variance, their loading
columns are orthogonal and ordered by how much of the channels they explain,
and their signs are fixed. So z1 is the main mode, z2 the next, and so on.

## Results

*(filled in from `uv run morrill kalman`)*
