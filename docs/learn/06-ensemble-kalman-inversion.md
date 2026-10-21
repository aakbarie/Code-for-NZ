# 06 · Ensemble Kalman inversion: physics and statistics, along the line

Note 05 ended at a wall. A model trained to copy the 2012 inversion can only
be as good as that inversion, and on the four lines nobody inverted
(L30070–L30100) it put the buried valley 15–40 m too shallow, with intervals
that missed every valley hole. The way past is to stop copying and invert the
data ourselves, keeping what the Kalman idea does well: carry knowledge from
one sounding to the next along the line.

This note builds that in two parts:
- **the physics:** what the bird should measure over a given layered earth
  ([`fdem.py`](../../src/morrill/analysis/fdem.py), about 60 lines of numpy);
- **the statistics:** how an ensemble of earths is nudged, sounding by
  sounding, until it agrees with the data
  ([`enki.py`](../../src/morrill/analysis/enki.py)).

## Part 1: the physics

### What the bird measures

The transmitter coil carries an alternating current, so it acts as a magnetic
dipole. Its field (the **primary**) induces eddy currents in the ground; their
own field (the **secondary**) reaches the receiver coil 7.9 m away (9.0 m for
the coaxial pair). The system reports the secondary as a fraction of the
primary, in parts per million, split into two parts:
- **in-phase (I):** in step with the transmitter current;
- **quadrature (Q):** a quarter cycle behind it.

A good conductor responds mostly in phase; a weak one mostly in quadrature.

How deep each frequency sees is set by the **skin depth**, δ ≈ 503 √(ρ/f)
metres, the depth at which the induced field has dropped by a factor of e:

| Frequency | δ at 10 ohm·m | δ at 40 ohm·m | δ at 150 ohm·m |
|---|---|---|---|
| 378 Hz | 82 m | 164 m | 317 m |
| 1,776 Hz | 38 m | 76 m | 146 m |
| 3,345 Hz (coaxial) | 28 m | 55 m | 107 m |
| 8,171 Hz | 18 m | 35 m | 68 m |
| 41,020 Hz | 8 m | 16 m | 30 m |
| 129,550 Hz | 4 m | 9 m | 17 m |

Under resistive gravel (about 150 ohm·m) even 8 kHz reaches 60 m, so the
valley's floor of Pierre Shale should be within reach.

### From a layered earth to ppm

For flat layers, the secondary field is an integral over horizontal
wavenumber λ (Ward & Hohmann, 1988). For the coplanar coils:

```
Hs/Hp = −r³ ∫ r_TE(λ) e^(−2λh) λ² J0(λr) dλ
```

The coaxial coils have the same form, with J0 replaced by J0 − J1/(λr) and a
factor ½.
- **h** is the bird height and **r** the coil separation.
- **e^(−2λh)** is the trip down to the ground and back up. It is why height
  matters so much: the response falls roughly as 1/h³.
- **J0** is a Bessel function: the coil geometry.
- **r_TE(λ)**, the reflection coefficient, holds all the geology. It comes
  from a recursion that starts in the bottom layer and works up, one line of
  code per layer (`reflection_te`).

Two checks show the integral is right:
1. **A perfect conductor** has r_TE = −1, and the integral reduces to the
   field of a mirror-image dipole 2h below the bird. The code matches that
   exact formula to 10⁻⁵ (`test_perfect_conductor_matches_the_image_dipole`).
2. **Random layered earths** agree with slow adaptive quadrature to within
   0.01 ppm; the noise is 5–40 ppm.

The integral is computed with the plain trapezoid rule in ln λ, 6 points per
decade (24 in all). That is enough because the integrand vanishes smoothly at
both ends; it is the same idea that makes "digital filters" work in EM codes.
One numpy call evaluates an ensemble of earths at once: about 2,700 earth
models per second per core.

### Does it reproduce the 2012 inversion's data?

Running EM1DFM's own resistivity models back through `fdem.py` and comparing
with the measured data tests everything at once: the coil geometry, the sign
conventions, the units and the height.

| Channel | Predicted / measured | Correlation |
|---|---|---|
| 1,776 Hz I, Q | 1.00, 0.98 | 0.999, 0.999 |
| 400 Hz Q | 1.00 | 0.999 |
| 8,171 Hz Q; 41 kHz I, Q | 0.98; 0.98, 0.96 | ≥ 0.996 |
| 129 kHz I, Q | 1.06, 1.03 | 0.995, 0.989 |
| **400 Hz I** | **0.83** | 0.990 |
| **coaxial 3,345 Hz I, Q** | **0.87, 0.88** | 0.996 |
| **8,171 Hz I** | **1.10** | 0.999 |

Most channels agree to within 1–5%. Three are off by 10–17%, consistently,
whatever the geology. Our own inversion, a different method, misses the same
three channels by the same amount and in the same direction (below). When two
different inversions can't fit a channel in the same way, the problem is in
the data, not the inversions: most likely a calibration error on those
channels.

## Part 2: the statistics

### The unknowns and the prior

At each sounding the unknowns are

```
θ = [log10 ρ of the 24 layers and the half-space, δh]
```

The layering is the same as EM1DFM's, so results can be compared layer by
layer. δh is a correction to the laser altimeter, which can read the
treetops instead of the ground.

The **prior** says what is plausible before seeing the data. It is learned,
not invented:
- **Mean and spread** of log10 ρ come from the induction logs, averaged into
  the model layers: about 40 ohm·m, with a spread of a factor of about 2.
- **Vertical correlation** comes from the logs too:
  `corr(Δz) = c + (1 − c) · exp(−Δz / L_v)`.
  - c is the share of variation that moves a whole profile together (one
    hole simply more resistive than another).
  - L_v is how far down a local change carries.
- **Holes are held out.** The 13 log holes within 100 m of the survey are
  left out, so the prior can't leak the answers the checks use. That leaves 19.

The logs pin down the short-range shape (correlation 0.87 at 1 m, 0.67 at
3 m). They do **not** pin down c. The 13 near holes suggest about 0.7, the 19
far ones about 0. With 6–19 holes per pair of layers, a correlation carries a
standard error of about 0.25. So c is chosen from the airborne data instead
(below), with L_v refitted to the logs for each choice.

**Along the line**, θ follows an AR(1) process:

```
θ_t = m + a (θ_{t−1} − m) + w_t,    a = exp(−Δ / L_h),    w_t ~ N(0, (1 − a²) Σ)
```

It is the discrete form of a Gaussian process with correlation
exp(−distance / L_h). The factor (1 − a²) is chosen so that the distribution
at every single sounding is exactly the prior N(m, Σ) (a test checks this).
L_h only controls how far knowledge carries along the line. With a = 0.99
this is the same dynamics as note 05's A matrix.

### The update: a Kalman filter for a nonlinear problem

The Kalman filter in note 05 assumed the observations were a *linear*
function of the state. EM physics isn't linear. The **ensemble** Kalman
filter gets around that by never writing the physics down as a matrix:
1. **Represent the uncertainty by N earths** (N = 400), drawn from the
   current distribution.
2. **Run the physics on all of them.** This gives N predicted data vectors.
3. **Measure covariances from the ensemble.**
   - C_θd: how each parameter co-varies with each channel across the
     ensemble. This is the physics' sensitivity, learned by example.
   - C_dd: how the channels co-vary with each other.
4. **Move every member** by the Kalman gain:
   ```
   θ_i ← θ_i + C_θd (C_dd + α R)⁻¹ (d_obs + √α ε_i − g(θ_i))
   ```
   - R is the data error, assumed to be √(floor² + (rel · |d|)²), with the
     floors from the 2015 paper's Table 1.
   - ε_i is fresh noise per member, so the ensemble keeps an honest spread.

**ES-MDA** (Emerick & Reynolds, 2013) does step 4 four times with the data
error inflated by α = 4. Many small steps follow the curvature of the physics
better than one big one. For a linear problem the result is exactly the
Gaussian posterior (`test_es_mda_gives_the_exact_posterior_for_a_linear_problem`).

**Along the line:**
- **Forecast:** push the ensemble one step with the AR(1) above.
- **Update:** apply ES-MDA with this sounding's data.
- **Smoother:** after the last sounding, an ensemble Rauch–Tung–Striebel
  smoother runs back along the line, so every sounding uses data from both
  sides, just as in note 05.
- **The anchor test:** for linear observations, the ensemble filter and
  smoother reproduce note 05's exact Kalman smoother, gaps included
  (`test_ensemble_filter_and_smoother_match_the_exact_kalman_smoother`).

**Soundings:** every 20th one is used (about 64 m apart), roughly one
footprint, so neighbouring data aren't near-copies of each other. Where the
bird flew above 60 m, the step is forecast but not updated.

### Two modes

- **Coupled:** the along-line filter and smoother above.
- **Independent:** every sounding starts from the same prior and borrows
  nothing from its neighbours. It is a per-sounding Bayesian inversion, the
  job Burke's MCMC did.

### Why the independent mode struggles

From the broad prior, ES-MDA can't fit the data well. The median misfit stays
at 2.0–2.3, whether it gets 4, 8 or 16 passes; the coupled mode reaches about
0.9. The ensemble update is Gaussian: it can only shift and stretch a cloud of
models. From a broad start, the cloud spans models whose responses differ
nonlinearly, and the linear step can't reach the answer. The coupled mode
starts each sounding from its neighbour's answer, so each update is small and
nearly linear. **Borrowing from the neighbours is what makes the ensemble
method work at all here**, not just a way of smoothing.

### Ensemble size

Fewer members understate the uncertainty. The width of the 80% interval for
layer 12 was 0.40 decades with 100 members, 0.45 with 200 and 0.54 with 400.
The final runs use 400. The search over settings uses 100, where only the
ranking matters.

### Choosing the settings: the marginal likelihood

Several settings are not knowable in advance:
- the along-line correlation length L_h;
- how much to widen the logs' spread;
- the whole-profile share c;
- the data error.

They are chosen the same way EM chose note 05's parameters, by the
**marginal likelihood**: the probability of the data under a setting, with
the earth integrated out. A Kalman filter computes it for free: at each
sounding, how well did the forecast predict the data before seeing them?
That is

```
log p(d_t | d_1 … d_{t−1}) = log N(d_t; mean of g(forecast), cov of g(forecast) + R)
```

summed along the line. The search runs on **line 30050**. It has no borehole
within 100 m, so tuning can't peek at the answers, and it is the line Burke's
MCMC run was set up for.

### Calibration

The one-gain-per-channel idea:
- A constant factor on a channel at every sounding can't be geology, which
  changes from sounding to sounding. So each channel gets a survey-wide gain:
  the median of measured / predicted.
- A factor common to *all* channels looks just like the bird flying a little
  lower, which δh absorbs. So the gains are only defined relative to each
  other, and their geometric mean is held at 1.

Iterating "invert, then re-estimate the gains" separates the channels into
two groups:
- **Stable from the first pass:**
  - the coaxial pair, measured about 12% high;
  - the 8.2 kHz in-phase, measured about 10% low.

  These are the channels the EM1DFM check flagged.
- **Drifting:** the 400 Hz in-phase and 140 kHz gains keep creeping, by about
  4% per round, because deep conductivity and bird height can trade off
  against them.

So the gains come from a single pass, from a model fitted to the uncorrected
data. On line 30050 that one pass raised the log marginal likelihood from
−20,272 to about −18,900, a large margin for the cost of 11 numbers.

## Results

*Pending: the first full run of `uv run morrill enki` is in progress.*
