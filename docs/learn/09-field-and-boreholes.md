# 09 · One surface across the block, and boreholes as data

Note 08 composed each flight line exactly, but on its own. The bedrock
surface doesn't stop at a flight line. The lines are about 200 m apart, and
along a line the evidence says log z_b changes by only about 0.05 decades
(12%) every 64 m. So each line knows a good deal about the next. And the
boreholes were only ever used to build the prior and to check the answer,
never as data.

This note composes in two more directions:
- **across lines,** so the block is one surface;
- **from boreholes,** so a logged depth informs the soundings around it.

It then takes one step down the chain toward groundwater: from the bedrock
posterior to the thickness of coarse sediment.

## Across lines: Gibbs sampling with whole lines as blocks

Exact composition of a 2D field is out of reach: on a grid of 80 depths, the
cost grows as 80 raised to the number of lines. But one line's whole surface,
**given** its neighbours' surfaces, is still a chain, and note 08 composes a
chain exactly.

[`bedrock_field.py`](../../src/morrill/analysis/bedrock_field.py) does this:
1. **Pick a line.** Hold its neighbours' current surfaces fixed.
2. **Compose it exactly** (forward pass), and draw its whole surface in one go
   (backward sampling).
3. **Move to the next line,** and sweep through all of them, many times.

This is a **Gibbs sampler whose blocks are whole lines**. It converges to the
exact posterior of the 2D model. On a tiny two-line field, the test compares
it with brute-force enumeration of every joint state
(`test_line_block_gibbs_converges_to_the_exact_2d_posterior`).

Two refinements:
- **Rao–Blackwellisation:** at each sweep, each line's exact conditional
  marginals are averaged, rather than counts of the draws. Same answer, much
  less noise.
- **The best field:** the same sweep in max-plus (each line's best path
  given its neighbours) climbs to a locally best field.

**The links across lines.** Each sounding is linked to its nearest sounding
on each neighbouring line, and to the tie lines where they cross. Across a
link, log10 z_b may change by a Gaussian amount whose variance grows with
distance, at the rate the along-line evidence chose:

```
variance = v · distance,   v = 2σ² / L_h
```

With note 08's choice (L_h = 24 km, σ = 0.65 decades), that is 0.083 decades
across 200 m. This assumes the surface changes as fast across lines as along
them (isotropy). The buried valley runs along the lines, so that may be
wrong, and the run checks how much it matters by halving and doubling v.

A practical note: the first version took over ten minutes per 600 sweeps,
because the general log-space composition pays a fixed overhead at every
step. For probabilities, the same composition can run in plain numbers, with
one 80 × 80 matrix–vector product per step, rescaled so nothing underflows
(`profunctor.Line`; `test_the_fast_linear_line_matches_the_log_space_chain`).

## Boreholes as data

A borehole is one more link, to the sounding nearest it:

```
log10 z_hole  ~  N( log10 z_b,  s0² + v · d )
```

- **v · d:** the surface's own change over the d metres between hole and
  sounding, the same rate as above.
- **s0:** everything else. A hole is a point while the bird averages about
  100 m, and a driller's contact is not quite an electrical one. s0 is fitted
  on the 15 calibration holes, by maximising how probable their depths are
  under the field's posterior. It is never fitted on the validation holes.

Two tests of what boreholes add:
1. **Strict:** the 15 calibration holes as data; predict the 21 validation
   holes, which the inversion never saw.
2. **Leave one out:** every other hole as data, the other 20 validation holes
   included; predict each validation hole in turn. This is how the map would
   be used in practice, with every hole that exists.

## From bedrock to coarse sediment

[`coarse.py`](../../src/morrill/analysis/coarse.py) composes one more
relation. At every grid depth, the sharp-contact fit gives the resistivity of
the five fill layers. Note 04's histogram model turns resistivity into
P(coarse / fine / bedrock).
- **Retrained for this inversion:** the histogram model must be trained on
  the inversion it is applied to, so here it is retrained on these fill
  layers against the logged lithology inside each one.
- **Scored honestly:** one hole is left out at a time, as in note 04.

For each sampled bedrock surface, the coarse sediment above it is

```
Σ over fill layers of P(coarse) × layer thickness
```

with the fill fitted at that surface's depth. Summed over the strips, that
gives a volume of sand and gravel, with uncertainty.

## Results

From `uv run morrill field`, then `uv run morrill coarse`.

### Across lines: a small gain

- **The field:** 3,783 links between lines. Across 200 m, log10 z_b may
  change by 0.083 decades (isotropic).
- **Sampling:** two chains of 300 sweeps took 1.2 minutes. Their medians
  agree to 0.001 decades at the median sounding and 0.07 at the worst 1%.
- **Against the boreholes:** the mean error falls from 3.72 m (lines alone)
  to 3.56 m. Halving or doubling the across-line variance gives 3.56 and
  3.62 m, so the isotropy assumption hardly matters.
- **In the valley, worse:** 1-H-97 moves from 38 to 33 m (bedrock at
  60.3 m). The figure shows why. The hole sits beside a steep valley wall,
  and its nearest sounding, 91 m away, is on the flank, where the surface
  rises 20 m in 300 m. Borrowing from the neighbouring line pulls the flank
  shallower still. Part of "the valley floor is too high" is a hole and a
  sounding that sit at different places on a wall.
- **Intervals, narrower still:** with information pooled across lines, the
  80% interval holds 10% of the holes, and the conformal factor rises to 6.9.

### Point against footprint: what the intervals were missing

Bedrock depth is rough at the scale of a single borehole. Pairs of holes
measure that directly. With 79 holes that are not validation holes, and
pairs closer than 400 m:

```
variance of the difference in log10 depth  =  0.0024 + 2.84 × 10⁻⁴ · d      (d in m)
```

| Distance apart | sd of the difference in log10 depth | As a factor |
|---|---|---|
| 0–100 m (27 pairs) | 0.14 decades | ±38% |
| 100–200 m (25 pairs) | 0.23 | ±70% |
| EM field, 64 m along the line | 0.047 | ±11% |

At the scale of single holes, the surface is roughly ten times rougher (in
variance) than the surface the bird sees. **A borehole measures a point; the
EM measures an average over about 100 m.** So an interval for the EM's
surface is not an interval for a borehole's depth.

Add the point-scale variability to the EM's interval, with nothing fitted to
the holes being scored:

| | Validation holes inside the 80% interval |
|---|---|
| 2D field, the EM's own interval | 10% |
| **+ point-scale variability, from borehole pairs** | **57%** |
| + the conformal factor of note 07 (lumps everything) | 76% |

Most of the "overconfidence" of notes 07 and 08 was this mismatch of scale,
not a failure of the EM model. The rest, from 57% toward 80%, is genuine
model error:
- **1-H-97**, on the valley wall;
- **1-G-96**, a Chadron contact the EM puts 8 m too high;
- **the shallowest holes** (0.6–1.5 m), below the model's 1 m floor.

**For volumes the distinction matters.** Point-scale roughness averages out
over an area, so stretching the surface by the conformal factor (6.9)
overstates the uncertainty of a volume. The point-scale variability belongs
to predictions at a single hole, not to totals.

### Boreholes as data: they barely move the map

The error of a borehole link is the point-scale variability above, not a
value fitted to the EM. Fitting the EM–hole disagreement alone would have put
0.23 decades on each hole, about the same figure for a different reason.

| | Mean error | Valley | Inside 80% |
|---|---|---|---|
| 2D field (+ point-scale variability) | 3.56 m | 19.0 m | 57% |
| + the 15 calibration holes as data | 3.53 m | 18.7 m | 57% |
| + every other hole as data (leave one out) | 3.56 m | 19.1 m | 57% |

A hole 50 m from a sounding is a ±35% measurement of a surface that the EM
already pins to about ±12%, so it gets little weight. Boreholes would matter
where the EM is biased and holes are dense. In this survey the bias is in the
valley, where there are only three deep holes, one of them on a wall.

### Coarse sediment: the chain stops being informative

Retrained on the grid inversion's fill layers (58 holes, 280 fill layers,
one hole left out at a time):

| | Score |
|---|---|
| Accuracy | 0.68 |
| Aquifer (coarse) yes or no | 0.68, against 0.67 for the 20 ohm·m rule |
| Coarse layers recognised | 99% |
| Fine layers recognised | **0%** |

The fill's resistivity doesn't separate silt and clay from sand at this
resolution, as note 04 found with EM1DFM. The model falls back to calling
the fill coarse, close to the 87% coarse share the training holes log. The
coarse-sediment volume, 1.09 km³ of the 1.25 km³ above bedrock, is that
share times the total, not a map of where the sand is.

### The volumes

| Under the ten lines' strips (about 45 km²) | km³ |
|---|---|
| Smooth model | 1.11 |
| Grid, lines alone (note 08) | 1.22 |
| **2D field: median [80%]** | **1.25 [1.24–1.26]** |
| Coarse sediment within it | 1.09 |
| *EM1DFM's six lines:* smooth / EM1DFM / field | 0.72 / 0.86 / 0.79 |

![Lines 30080 and 30090: each line alone, the 2D field and its 80% interval,
and each validation hole predicted from every other hole](../figures/field_lines.png)

## What this gives us

- **One surface for the block,** from an exact sampler of the 2D model, in
  about a minute. It is slightly more accurate than the lines alone.
- **The scale finding.** Holes 50 m apart differ by ±35% in depth to
  bedrock. Borehole pairs measure this independently, and it explains most
  of the gap between the EM's intervals and the boreholes, with nothing fitted
  to the holes being scored.
- **A realistic view of boreholes as data:** at this density, and with this
  roughness, they hardly move a map the EM already constrains.
- **Where the chain toward groundwater stops:** resistivity here can't tell
  fine sediment from coarse. Hydraulic conductivity needs aquifer tests or
  another measurement, and heads need water levels, wells, the river and the
  canals.

## Questions to think about

1. Gibbs with whole lines as blocks mixes far faster than Gibbs with single
   soundings. Why? What happens to single-sounding Gibbs when neighbours are
   strongly coupled?
2. s0 bundles a footprint effect and a definition effect. How could you
   separate them with the data you have?
3. The across-line coupling assumes isotropy, but the buried valley runs
   north–south along the lines. Which way would you expect the error to go,
   and how would you estimate the right anisotropy?
