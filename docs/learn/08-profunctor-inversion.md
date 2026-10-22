# 08 · The inversion as a profunctor

Note 07 found a weakness that no setting could fix. Where the data allow two
depths to bedrock, a shallow one and a deep one, the ensemble keeps only one,
and its random seed decides which. The reason is structural: an ensemble
Kalman update moves a cloud of models as a Gaussian, and a Gaussian has one
bump.

This note changes the representation instead of the settings.

## Why an inversion is not a function

The physics is a function: one earth gives one set of data. An inversion runs
the other way, and the other way is not a function, because many earths fit
the same data. Every method so far has pretended otherwise in some way:
- **EM1DFM** returns one earth.
- **Note 05's machine learning** returns one depth per sounding.
- **Note 06 and 07's ensembles** return one Gaussian.

The mathematical object that fits is a **profunctor**. Here that means a table
of compatibilities between two spaces, together with a rule for composing two
tables through a space they share:

```
(Q ∘ P)[a, c] = ⊕_b  P[a, b] ⊗ Q[b, c]
```

That is matrix multiplication with the + and × replaced by two operations
(⊕, ⊗) of your choosing, a **semiring**. Composition is associative and has an
identity (`test_composition_is_associative_with_an_identity`), so spaces and
tables form a category. That is what lets the pieces snap together: soundings
along a line, and later lines across the block and the borehole–EM–groundwater
chain.

| Entries of the table | It says | ⊕, ⊗ | Composing along a flight line gives |
|---|---|---|---|
| true / false | which earths fit | or, and | which surfaces are consistent with every sounding |
| a probability, in logs | how likely each earth is | log-sum-exp, + | the posterior at every sounding: **forward–backward** |
| a log-probability as a cost | how bad each earth is | max, + | the single most probable surface: **Viterbi** |

The Kalman filter and RTS smoother of notes 05–07 are the second row with
everything Gaussian. Drop the Gaussian and lay the space out on a grid, and
the same composition is exact for any shape of posterior, including two bumps.

The algebra is in [`profunctor.py`](../../src/morrill/analysis/profunctor.py).
It has no geophysics in it: `compose`, `chain` (both semirings), `best_path`,
`sample_paths` and `reversible_kernel`.

## What goes on the grid

A grid only represents a posterior faithfully if its spacing is finer than
how sharply the data pin each number. The Fisher information of the data at
typical Morrill earths says how sharply that is:

| Earth | sd of log fill ρ | sd of log rock ρ | sd of log z_b | sd of bird height |
|---|---|---|---|---|
| 40 on 7 ohm·m, contact at 6 m | 0.20 | 0.024 | 0.054 | 1.0 m |
| 40 on 7 ohm·m, at 30 m | 0.033 | 0.066 | 0.041 | 0.43 m |
| 79 on 8 ohm·m, at 50 m | 0.029 | 0.10 | 0.038 | 0.52 m |
| 20 on 6 ohm·m, at 12 m | 0.056 | 0.037 | 0.060 | 0.41 m |

A grid fine enough for all eight of note 07's numbers would need over a
million cells per sounding. So:
- **Only the depth to bedrock goes on the grid:** 80 values from 1 to 150 m,
  0.027 decades apart, about half a posterior standard deviation.
- **The other seven numbers ψ are integrated out,** at every grid depth of
  every sounding:

  ```
  L_t(z) = ∫ p(d_t | z, ψ) p(ψ) dψ
  ```

  This is itself a composition, through the space of ψ. It is done by a
  Laplace approximation around a Levenberg–Marquardt fit.
  - Each fit starts from the same depth at the previous sounding, so four
    steps are enough.
  - Every tenth sounding it also starts from the prior mean, and keeps
    whichever fits better, so a fit can't drift into a poor solution.
  - For a linear forward model the Laplace step is exact
    (`test_laplace_is_exact_for_a_linear_forward_model`).

The earth and the prior are note 07's, unchanged: five fill layers in fixed
proportions of z_b, a bedrock half-space and a bird-height correction, with
the prior from held-out boreholes. There are two differences:
1. **The Laplace step,** the one approximation.
2. **ψ isn't coupled along the line.** Only z_b is.

## Along the line

The depth to bedrock follows note 07's AR(1) prior, now as a table from one
sounding's grid to the next. Discretizing a Gaussian conditional on a grid
shifts the prior slightly. A Metropolis–Hastings correction removes that, so
the prior is kept exactly at every sounding
(`test_the_line_kernel_keeps_the_prior_exactly`).

From there it's all composition:
- **sum-product** gives the posterior at every sounding, its quantiles, and
  how many separate answers it holds;
- **max-plus** gives the single most probable bedrock surface;
- **sampling backwards** through the forward messages gives exact joint draws
  of whole surfaces, for the aquifer volume.

For a linear-Gaussian line the grid posterior matches the exact Kalman
smoother to 0.01 in mean and standard deviation
(`test_a_linear_gaussian_line_matches_the_exact_kalman_smoother`). Brute force
over every path confirms both semirings on small lines.

## Results

From `uv run morrill profunctor`, which also runs `aquifer.py` on the sampled
surfaces.

### The settings, chosen for free

The evidence at each sounding, L_t(z), depends neither on the along-line
length L_h nor on the width of the depth prior. So after one 5.4-minute pass
of fits (0.36 core-seconds per sounding), any choice of either can be scored
over all 12 lines by composition alone, in seconds:

| L_h | Depth spread × | log evidence | Change in log z_b allowed per 64 m step |
|---|---|---|---|
| **24 km** | **1.5** | **−206,235** | **0.047 decades** |
| 12 km | 1.0 | −206,249 | 0.044 |
| 6 km | 0.75 | −206,254 | 0.047 |
| 6 km | 1.0 | −206,277 | 0.063 |
| 3 km (note 07's choice) | 1.0 | −206,535 | 0.089 |
| 1 km | 1.5 | −208,114 | 0.22 |

The top three are a ridge, not a peak. They allow almost the same change in
depth from one sounding to the next, about 0.05 decades (12%), and that is
what the data pin down; L_h on its own they barely constrain. The run uses
the top row.

### The trap on line 30090

Note 07 said the data on line 30090 allow two answers, a shallow contact and
a deep one, and that the ensemble keeps whichever its seed lands on. **That
was wrong.**

| Hole | Bedrock | Note 07 ensemble (pooled) | Grid median | Grid: P(z_b > 8 m) |
|---|---|---|---|---|
| 11AK-94 | 0.6 m | 12.3 m | 2.9 m | 10⁻⁹ |
| 14-K-94 | 1.5 m | 12.6 m | 3.8 m | 3 × 10⁻⁶ |
| 15-K-94 | 3.7 m | 12.1 m | 4.2 m | 10⁻⁵ |
| 11-K-94 | 4.6 m | 18.2 m | 3.1 m | 10⁻⁷ |

The grid evaluates every depth at every sounding, and it gives the deep
contact no real probability: one in a hundred thousand or less. The deep
answer wasn't in the data. It was a trap the Gaussian update fell into and
couldn't climb out of. Only 7 of 3,577 soundings (0.2%) hold two separate
answers, and none of them is at a validation hole.

### Against the boreholes

| Holes | n | Grid | Note 07 ensemble, pooled | Smooth model (note 06) | EM1DFM |
|---|---|---|---|---|---|
| Inverted lines | 10 | **1.5 m** | 2.6 m | 1.7 m | 2.1 m |
| Uninverted lines | 11 | **5.7 m** | 8.9 m | 8.4 m | — |
| Buried valley | 2 | 17.0 m | **12.7 m** | 20.7 m | — |
| All | 21 | **3.7 m** | 5.9 m | 5.2 m | — |

- **Where bedrock is shallow,** the grid is the most accurate inversion so
  far, better than EM1DFM.
- **On the uninverted lines,** the error falls by a third, mostly because
  line 30090 is no longer trapped.
- **In the valley,** it does worse at one hole: 1-H-97, 91 m from its
  sounding, comes out at 38 m [34–43] against the ensemble's 47 m. At 2-J-97
  both give 47 m. At 1-J-96 (59.7 m), a calibration hole, the grid gives
  55 m [48–63].
- **The data fit is better too:** the median misfit at the best surface is
  0.21, against 0.31 for the ensemble.

### Exact inference, still overconfident intervals

The errors halve in log depth (rms 0.23 decades, against 0.43), but the
intervals shrink too (typical half-width 0.06 decades). So:
- **Raw:** 3 of 21 validation holes fall inside the 80% interval (14%), and
  4 of the 15 calibration holes.
- **Recalibrated:** the split-conformal factor is k = 4.08. After
  stretching, 16 of 21 are inside (76%).

**This separates two failures that note 07 had mixed together:**
1. **An inference failure:** the ensemble got trapped. Exact composition on a
   grid fixes it, and accuracy improves.
2. **A model failure:** a 1D, two-unit earth, ψ uncoupled along the line,
   and a borehole that is a point while the bird sees about 100 m. Exact
   inference can't fix that, because the posterior is exact *for the wrong
   model*. Only checking against boreholes the fit never saw can.

### What it means for the aquifer

| | Volume of aquifer material under the ten lines (km³) |
|---|---|
| Smooth model | 1.11 |
| **Grid, sharp contact: median (80%)** | **1.22 (1.22–1.24)** |
| Grid, recalibrated 80% | 1.30–1.36 |
| Note 07 ensemble: pooled / single start | 1.41 / 1.52 |
| *EM1DFM's six lines only:* smooth / EM1DFM / grid | 0.72 / 0.86 / 0.77 |

Note 07 said the choice of inversion moves the volume by 27–37%. Most of
that gap was the trap: the ensemble put deep contacts under shallow bedrock
on several lines. With exact inference the sharp contact gives **10% more**
material than the smooth model. That is still more than any single run's
stated uncertainty, but a third of what note 07 claimed.

![Lines 30090 and 30070: the grid posterior of the depth to bedrock at every
sounding, the most probable surface, note 07's ensemble interval, and the
boreholes](../figures/profunctor_lines.png)

![Depth to bedrock from the grid on every line, and the few soundings whose
posterior holds two answers](../figures/profunctor_map.png)

## What this gives us

- **A better answer:** 3.7 m mean error at the 21 validation holes, the best
  so far, in five minutes on four cores.
- **A correction:** the "two answers" of note 07 were a weakness of the
  ensemble, not of the data.
- **A clean split** between errors of inference, which exact composition
  removes, and errors of the model, which only held-out boreholes reveal.
- **Settings chosen by evidence for free,** because the expensive part (the
  per-sounding evidence) doesn't depend on them.
- **A small algebra** (`profunctor.py`) that the next steps can reuse:
  lines across the block, boreholes as a second set of observations, and the
  chain from earth to groundwater heads.

## Questions to think about

1. The Kalman smoother and Viterbi are the same composition in two
   semirings. What does a third semiring, (max, ×) on probabilities rather
   than logs, compute? Is it different from max-plus?
2. Only z_b is carried along the line; ψ is integrated out independently at
   each sounding. What information does that throw away, and where would you
   expect it to matter?
3. The grid needs a spacing finer than the posterior width. Why does that
   rule out grids for most geophysical inversions, and what does the Laplace
   step buy back?
