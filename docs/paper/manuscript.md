---
title: "Ensemble Kalman inversion of helicopter electromagnetic data along flight lines: mapping the base of a buried-valley aquifer, with uncertainty, at a fraction of the cost of MCMC"
author:
  - Akbar Esfahani
date: "Draft, October 2026"
---

**Author:** Akbar Esfahani, [affiliation], [address]. Correspondence: [email]. ORCID: [ORCID].

**Target journal:** [to be chosen]

**Keywords:** airborne electromagnetics; ensemble Kalman methods; Bayesian inversion; groundwater; buried valley; uncertainty quantification

# Abstract

ABSTRACT

# 1 Introduction

Helicopter frequency-domain electromagnetic (HEM) surveys measure the
electrical response of the ground every few metres along flight lines, over
areas of hundreds of square kilometres. In alluvial aquifers the response is
dominated by the contrast between resistive sand and gravel and conductive
clay, silt and shale. That contrast makes HEM data a direct route to the
geometry hydrogeologists need, above all the base of the aquifer.

Turning the measurements into resistivity requires an inversion, one sounding
at a time. Deterministic inversions such as EM1DFM (Farquharson et al. 2003)
fit a smooth layered model to each sounding. They are fast, but they give one
answer, with no statement of how well it is determined. Bayesian
alternatives answer that question. Trans-dimensional Markov chain Monte Carlo
(MCMC; Malinverno 2002; Minsley 2011; Minsley et al. 2021) samples models
whose number of layers is itself unknown, and so describes non-uniqueness
directly. The price is computation. With the 2012 version of one such code,
a single test sounding took about 200,000 forward-model evaluations and
24 minutes (section 4.6). A survey of 70,000 soundings would need years of
single-core time.

An earlier study of the same survey proposed a hybrid route (Friedel et al.
2015): invert a subset of soundings numerically, train a machine-learning
model on the inverted profiles, and let it predict resistivity everywhere
else in near real time. The idea is attractive, but a model trained to
reproduce an inversion inherits the inversion's limitations. It can be no
better than its training labels, and its uncertainty describes how well the
inversion was copied, not how well the ground is known (section 4.5).

This paper takes a different route to the same goal. The physics is kept and
made cheap with statistics: an ensemble Kalman inversion (Evensen 2009;
Iglesias et al. 2013; Emerick & Reynolds 2013) runs sounding by sounding
along each flight line. Each sounding's posterior becomes, through a
Gaussian-process prior along the line, the next sounding's prior. The
contributions are:

1. **A one-dimensional HEM forward model** written from first principles and
   tested against analytical and numerical references. It reproduces the
   data from an independent inversion's models to 1–5% on nine of twelve
   channels. The remaining three expose a calibration error in the survey
   data.
2. **An along-line ensemble Kalman inversion.** It combines ensemble smoother
   updates (ES-MDA) at each sounding with an ensemble Rauch–Tung–Striebel
   smoother. Its prior is learned from borehole induction logs held out of
   validation, and its hyperparameters are chosen by marginal likelihood.
3. **Two parameterizations, compared against boreholes.** A smooth layered
   model is compared with one in which the depth to bedrock is an explicit,
   sharp contact. The comparison shows why smooth models place the floor of
   a resistive buried valley too high, and what fixes it.
4. **A cost comparison with MCMC** using the original MCMC code, including a
   measured account of the speed-ups made to it in 2012.

# 2 Study area and data

## 2.1 The Morrill block

**Setting.** The study area is the Morrill block of the North Platte River
valley, Scotts Bluff County, western Nebraska. Quaternary alluvium and dune
sand overlie Tertiary siltstones and sandstones (Brule Formation and Chadron
Formation of the White River Group), which rest on Cretaceous Pierre Shale.
These units make up the primary surficial aquifer. A buried paleovalley,
filled with sand and gravel, is cut through the Tertiary rocks to the Pierre
Shale at about 60 m depth along easting 588 km. The water-bearing units and
the base of the aquifer were mapped regionally from airborne EM by Abraham et
al. (2012).

**The survey.** It was flown by Fugro Airborne Surveys with the RESOLVE
system, 15–23 June 2008, for the North Platte Natural Resources District
(Fugro Airborne Surveys 2008).
- **Instrument:** five horizontal coplanar coil pairs (378, 1,776, 8,171,
  41,020 and 129,550 Hz; separation 7.9 m) and one vertical coaxial pair
  (3,345 Hz; 9.0 m). Each records in-phase and quadrature response in ppm.
- **Sampling:** 10 Hz, about 3.2 m between soundings, with a laser
  altimeter giving the bird height.
- **Morrill block coverage:** ten north–south lines about 200 m apart and
  22.7 km long, plus two east–west tie lines; 71,470 soundings in all.

## 2.2 Boreholes

The boreholes are 130 test holes of the Conservation and Survey Division,
University of Nebraska–Lincoln, with three kinds of record:
- **driller's lithology** (all 130 holes; 1,173 intervals);
- **formation picks** (102 holes), from which the depth to bedrock follows
  as the top of the first Tertiary or Cretaceous unit, with 100 holes
  reaching bedrock;
- **induction-resistivity logs** (32 holes; 37,394 readings at 3 cm
  spacing).

The logs were averaged into the model layers in conductivity, because
current flows through thin beds in parallel.

## 2.3 A database, and what building it found

**The database.** All inputs were loaded into a single relational database
(DuckDB). A manifest records every source file with checksums, and eleven
automated checks run on every build. Building it brought several data
problems to light:
- **The lithology files' elevation column** gives the midpoint of each
  interval, not its top.
- **The survey's heights** refer to the WGS84 ellipsoid, while borehole
  collars are orthometric. The offset, 14.54 m (median of 30 holes within
  100 m of a sounding; spread 0.9 m), was applied before comparing them.
- **An existing deterministic inversion** (EM1DFM; 24 layers over a
  half-space to 125 m) covers only the first six lines (41,330 soundings).
  The four eastern lines, which cross the buried valley, and the tie lines
  were never inverted.

**Validation and prior holes.** Holes within 100 m of a sounding are used
only for validation: 21 holes with bedrock picks and 12 with induction logs.
Every prior quantity is estimated from the other holes.

# 3 Methods

## 3.1 Forward model

For a magnetic dipole above horizontally layered ground (quasi-static, non-magnetic layers), the secondary field normalized by the primary is a Hankel transform over the horizontal wavenumber $\lambda$ (Ward & Hohmann 1988). For horizontal coplanar coils at height $h$ and separation $r$,

$$\frac{H_s}{H_p} = -r^3 \int_0^\infty r_{TE}(\lambda)\, e^{-2\lambda h}\, \lambda^2 J_0(\lambda r)\, d\lambda,$$

and for vertical coaxial coils $J_0(\lambda r)$ is replaced by $\tfrac12[J_0(\lambda r) - J_1(\lambda r)/(\lambda r)]$, with the sign convention of the survey data (positive over a conductor). The reflection coefficient $r_{TE} = (\lambda - \hat u_1)/(\lambda + \hat u_1)$ follows from the upward recursion $\hat u_j = u_j(\hat u_{j+1} + u_j \tanh u_j t_j)/(u_j + \hat u_{j+1}\tanh u_j t_j)$, with $u_j = (\lambda^2 + i\omega\mu_0\sigma_j)^{1/2}$ and layer thickness $t_j$.

The integral is evaluated with the trapezoid rule in $\ln\lambda$ at six points per decade over $10^{-4}$–$1$ m$^{-1}$ (24 points). Because the integrand vanishes smoothly at both ends of that interval, the rule converges very fast. It matches the closed-form image-dipole solution for a perfect conductor to within $10^{-5}$ (relative) at heights of 15–60 m, and adaptive quadrature for random layered earths to within 0.01 ppm, against noise of 5–40 ppm.

The code is vectorized over models, with one array call per ensemble, and evaluates about 2,700 earth models per second per core.

## 3.2 Prior

**Smooth layered model.** The unknowns at each sounding are

$$\theta = [\log_{10}\rho_1,\dots,\log_{10}\rho_{25},\ \delta h],$$

the resistivities of the same 24 layers and half-space as EM1DFM, plus a correction $\delta h$ to the laser altimeter height, with prior $\delta h \sim N(0, 2^2)$ m.

The prior on $\log_{10}\rho$ is Gaussian:
- **Mean and spread** come from the 19 induction logs held out of validation: mean 1.60 (about 40 ohm·m), standard deviation 0.32. The spread is multiplied by a factor chosen by marginal likelihood (section 3.4).
- **Vertical correlation** follows $\mathrm{corr}(\Delta z) = c + (1-c)\exp(-\Delta z/L_v)$, between layer centres $\Delta z$ apart.

The logs constrain the short-range correlation well: 0.87 at 1 m and 0.67 at 3 m. They do not constrain the share $c$ of variance common to the whole profile: estimates range from about 0 to 0.7 between subsets of holes, with standard errors near 0.25. So $c$ is chosen by marginal likelihood, and $L_v$ is refitted to the logs for each choice.

**Sharp-contact model.** Here the depth to bedrock $z_b$ is itself a state variable:

$$\theta = [\log_{10}\rho^{\mathrm{fill}}_1,\dots,\log_{10}\rho^{\mathrm{fill}}_5,\ \log_{10} z_b,\ \log_{10}\rho_{\mathrm{rock}},\ \delta h].$$

The fill between the surface and $z_b$ is divided into five layers in fixed proportions of $z_b$ (5, 10, 15, 30 and 40%), so every ensemble member has the same structure but its own contact depth. Below $z_b$ is a bedrock half-space. The priors are again learned from held-out holes:
- $\log_{10} z_b$: from the 78 holes that reached bedrock;
- $\log_{10}\rho$ of fill and bedrock: from induction logs above and below the logged contact.

Because test holes cluster where bedrock is shallow, the depth prior is centred shallow (median 6 m). Its spread is widened by a factor chosen by marginal likelihood.

## 3.3 Ensemble Kalman inversion along a flight line

**Along-line prior.** Along a line, $\theta$ follows a first-order autoregressive process, the discrete form of an Ornstein–Uhlenbeck (exponential-covariance) Gaussian process:

$$\theta_t = m + a_t(\theta_{t-1} - m) + w_t,\qquad a_t = e^{-\Delta_t/L_h},\qquad w_t \sim N\!\left(0,(1-a_t^2)\Sigma\right),$$

where $\Delta_t$ is the distance from the previous sounding. The marginal distribution at every sounding is the prior $N(m,\Sigma)$; the correlation length $L_h$ sets only how far information carries along the line.

**Update at each sounding.** An ensemble of $N$ models is updated with ES-MDA (Emerick & Reynolds 2013). The Kalman update

$$\theta_i \leftarrow \theta_i + C_{\theta d}\,(C_{dd} + \alpha R)^{-1}\,(d + \sqrt{\alpha}\,\varepsilon_i - g(\theta_i))$$

is applied $N_a = 4$ times with the data covariance inflated by $\alpha = N_a$. The covariances $C_{\theta d}$ and $C_{dd}$ are estimated from the ensemble, and $g$ is the forward model. Data errors are $\sigma = (\sigma_{\mathrm{floor}}^2 + (0.05\,|d|)^2)^{1/2}$, with the channel floors of the survey specification (5–40 ppm).

**Backward pass.** After the last sounding, an ensemble Rauch–Tung–Striebel smoother (Rauch et al. 1965; Raanes 2016) runs back along the line, so that each estimate uses data on both sides.

**Settings.**
- Every 20th sounding is used (about 64 m apart, roughly one footprint).
- Soundings flown above 60 m are forecast but not updated (1.3% of the survey).
- The ensemble has $N = 400$ members.

Two tests fix the implementation:
1. For a linear observation operator, ES-MDA reproduces the analytical Gaussian posterior.
2. The ensemble filter and smoother reproduce the exact Kalman filter and RTS smoother of an independent linear state-space implementation, including a gap in the observations.

A "sounding-by-sounding" mode, in which every sounding starts from the same prior, serves as a per-sounding Bayesian baseline (the task MCMC performs).

## 3.4 Hyperparameters and calibration

**Hyperparameters.** The along-line correlation length $L_h$, the widening of the prior spreads, and (smooth model) the whole-profile share $c$ are chosen by maximum marginal likelihood. The ensemble filter evaluates $\log p(d_t \mid d_{1:t-1})$ at each sounding from the forecast ensemble, and these are summed along the line. The search uses line 30050, which has no borehole within 100 m, so it cannot draw on validation data.

The relative data error is not chosen this way. The likelihood treats soundings as conditionally independent, so it rewards ever smaller errors and cannot see model errors that repeat from sounding to sounding: the 1D assumption, calibration, and the difference in support between a borehole log and the bird's footprint. The relative error is therefore fixed at 5%, the level at which the independent EM1DFM models miss the data on most channels (section 4.1). The sensitivity to that choice is reported in section 4.4.

**Calibration.** Systematic misfit in three channels (section 4.1) motivated one survey-wide gain per channel, estimated as the median ratio of measured to predicted data after an inversion of line 30050. A gain common to all channels is indistinguishable from a change in bird height, so the geometric mean of the gains is fixed at 1.

Iterating the estimate does not converge for every channel:
- **The coaxial and 8.2 kHz in-phase gains** settle in the first pass.
- **The 400 Hz in-phase and 129 kHz gains** keep drifting, because deep conductivity and bird height trade off against them.

A single pass is therefore used.

## 3.5 Validation and cost

**Validation.**
- **Bedrock depth:** the estimate at the nearest sounding with data, within 100 m, is compared with the borehole depth to bedrock.
- **Smooth model:** the depth to conductor (top of the first layer below 1.8 m with $\rho < 20$ ohm·m, the binary split of Friedel et al. 2015) is compared.
- **Sharp model:** $z_b$ itself is compared.
- **Scores:** mean absolute error of the median, and the fraction of holes whose depth lies inside the 80% interval. For the smooth model, the interval counts as containing the depth if it reaches the model layer that holds it.
- **Resistivity:** the smooth model is also compared with the induction logs and with EM1DFM.

**Cost.** Run time is compared with the trans-dimensional MCMC code of Minsley (2011) as it existed in 2012, with a run log recorded then and benchmarks rerun here in GNU Octave 8.4 (section 4.6).

# 4 Results

## 4.1 Forward model and survey calibration

**The forward model against an independent inversion.** EM1DFM's resistivity models for the six inverted lines were run through the forward model and compared with the measured data (Table 1). Nine channels agree to within 1–5%, with correlations of 0.989–0.999. Three are off by 10–17%, consistently, whatever the geology: the 378 Hz in-phase, both coaxial channels and the 8,171 Hz in-phase.

**Table 1.** Data predicted from EM1DFM's models by the forward model, against the measured data (six inverted lines). Median ratio and correlation per channel.

| Channel | Predicted / measured | Correlation |
|---|---|---|
| 1,776 Hz I, Q | 1.00, 0.98 | 0.999, 0.999 |
| 378 Hz Q | 1.00 | 0.999 |
| 8,171 Hz Q; 41 kHz I, Q | 0.98; 0.98, 0.96 | ≥ 0.996 |
| 129 kHz I, Q | 1.06, 1.03 | 0.995, 0.989 |
| **378 Hz I** | **0.83** | 0.990 |
| **coaxial 3,345 Hz I, Q** | **0.87, 0.88** | 0.996 |
| **8,171 Hz I** | **1.10** | 0.999 |

**The same channels, from a different inversion.** The ensemble inversion, fitted to the uncorrected data on line 30050, misses the same channels in the same direction. One pass of the gain estimate (section 3.4) gives:
- coaxial in-phase and quadrature: 1.12 and 1.11 (measured high);
- 8,171 Hz in-phase: 0.90 (measured low);
- the other nine channels: 0.93–1.03.

When two inversions of different design cannot fit the same channels in the same way, the error lies in the data. The most likely cause is a calibration error on those channels. Applying the gains raised the log marginal likelihood of line 30050 from −20,272 to about −18,900, for the cost of eleven numbers. All results below use the corrected data.

## 4.2 Smooth layered model

**Settings.** Of 27 combinations searched on line 30050, the marginal likelihood chose:
- $L_h = 3$ km;
- prior spreads widened by 1.5 (one standard deviation is a factor of 3 in resistivity);
- $c = 0$, in the seven best combinations: given the airborne data, the deep layers are better left independent of the shallow ones. $L_v$, refitted to the logs with $c = 0$, is 9.1 m.

The scores of the leading settings are close (log marginal likelihoods −18,767, −18,799 and −18,803 for $L_h$ = 3, 6 and 1.5 km). The long correlation length mostly says that soundings 64 m apart tell nearly the same story.

**Run and data fit.** All 12 lines (3,577 soundings) were inverted in 9.9 min on four cores, or 0.67 core-seconds per sounding. The median mean squared normalized residual per sounding is 0.53, and 90% of soundings are below 0.95, so the data are fitted to within the assumed error. The median height correction is −0.2 m; 80% of soundings fall between −2.1 and +1.6 m.

**Against EM1DFM** (six inverted lines, 1,999 soundings):
- **resistivity:** the two agree to about 0.15 decades (40%) through the top 60 m (median absolute difference 0.14–0.15 decades), and 0.22 decades below. EM1DFM's value lies inside the ensemble's 80% interval for 66–86% of layer values, depending on depth.
- **depth to conductor:** median absolute difference 2.4 m, with no bias; EM1DFM's value lies inside the 80% interval at 83% of soundings.

**Against the boreholes** (Table 2). Where bedrock is shallow, on the inverted lines, the along-line ensemble is more accurate than EM1DFM (mean absolute error 1.7 m against 2.1 m), and its intervals contain the borehole depth at 9 of 10 holes. Run sounding by sounding from the same prior, the error rises to 2.7 m and coverage falls to 40%. On the four uninverted lines the error is 8.4 m. At the two buried-valley holes it is 21 m: the medians are 42 and 36 m, against 58.5 and 60.3 m in the boreholes, and neither interval reaches the logged depth.

**Table 2.** Depth to conductor (smooth model) against borehole depth to bedrock, at the 21 holes within 100 m of a sounding with data. Mean absolute error of the median, and fraction of holes inside the 80% interval.

| Holes | n | Along the line | Sounding by sounding | EM1DFM |
|---|---|---|---|---|
| Inverted lines | 10 | **1.7 m / 90%** | 2.7 m / 40% | 2.1 m |
| Uninverted lines | 11 | 8.4 m / 27% | 8.7 m / 55% | — |
| Buried valley (2-J-97, 1-H-97) | 2 | 21 m / 0 of 2 | 23 m / 1 of 2 | — |

**Why the valley floor is too high.** The profiles at the valley holes show the cause (Fig. 1). At 1-H-97 the inversion finds a strong conductor (3.6–7.5 ohm·m) from 63.6 m, the first layer top below the logged shale, so the shale is seen at about the right depth. It also puts 10 ohm·m at 36–42 m, where the driller logged sand and gravel. At 2-J-97 the resistivity falls gradually, to 11–22 ohm·m from 48 m and 3.4 ohm·m from 84 m. The induction log there reads 30–47 ohm·m down to 55 m, and clay begins at 58.5 m. Three things combine:
1. **The fill is not very resistive.** It logs at 30–80 ohm·m, only a factor of two above the 20 ohm·m threshold.
2. **The prior is smooth.** A correlated Gaussian cannot make a sharp step; it ramps down to the shale over 10–20 m.
3. **The rule takes the first layer below 20 ohm·m,** so it fires near the top of the ramp.

**Against the induction logs** (12 holes, 160 layer values), the rank correlation is 0.58, against 0.67 for EM1DFM where it exists. The ensemble reads 0.13 decades (35%) more resistive than the logs, and the log lies inside the 80% interval for 44% of values. That fraction is not a fair test of the interval, which describes the average over a footprint of about 100 m rather than the metre around a borehole. Both inversions read 25–35% more resistive than the logs throughout.

![Fig. 1. Line 30070, smooth model: median resistivity with poorly resolved cells shaded grey; the width of its 80% interval; depth to conductor with its interval and the boreholes; data misfit.](../figures/enki_line.png)

![Fig. 2. Line 30060: EM1DFM, the ensemble median, and the width of its 80% interval.](../figures/enki_vs_em1dfm.png)

## 4.3 Sharp-contact model

SHARP

## 4.4 Sensitivity to the data error

The marginal likelihood prefers ever smaller relative data errors, down to the channel floors: 1% scores −17,727 on line 30050, against −18,767 for 5%. At 1% the data are still fitted to within the error (median misfit 0.82), so the measurement noise is indeed small. The errors that matter are of another kind: the earth is not one-dimensional, the calibration is imperfect, and the bird averages over about 100 m where a log averages over a metre. All three repeat from one sounding to the next, which a likelihood that treats soundings as independent cannot see. Table 3 shows the trade-off for the smooth model.

**Table 3.** Smooth model at 1% and 5% relative data error, scored against boreholes and logs.

| Data error | Depth error, inverted lines | Inside 80%, inverted lines | Inside 80%, uninverted lines | Logs: rank correlation, inside 80% |
|---|---|---|---|---|
| 1% (likelihood's choice) | 1.6 m | 70% | 45% | 0.49, 36% |
| 5% (used) | 1.7 m | 90% | 27% | 0.58, 44% |

Neither setting is better everywhere: 5% is better on the inverted lines and the logs, and worse on the uninverted lines. Neither finds the valley floor.

## 4.5 A learned surrogate as a baseline

To test the premise of the hybrid approach, a gradient-boosting model (Friedman 2001) was trained to reproduce EM1DFM's depth to conductor from the measured channels and bird height. This is a stand-in in the spirit of Friedel et al. (2015), not a replication of their method. Two kinds of input were compared: the raw channels, and four along-line features from a linear state-space model fitted by expectation–maximization (Shumway & Stoffer 1982), with bird height as an explicit input.

**It copies the inversion well.** With each of the six inverted lines held out in turn (39,886 soundings), the raw channels reproduce EM1DFM's depth to conductor with a mean absolute error of 2.3 m ($R^2$ = 0.93). The state-space features do slightly worse (2.9 m, $R^2$ = 0.92), because the inversion is itself a sounding-by-sounding function of the raw channels.

**It inherits the inversion's limits.** On the four uninverted lines, both inputs place the deepest predictions exactly where the valley holes are, so the valley is detected. But the predictions there are 15–40 m too shallow:
- 2-J-97 (58.5 m): 44 m from the raw channels, 45 m from the features;
- 1-H-97 (60.3 m): 37 and 33 m;
- 1-J-96 (59.7 m): 37 and 20 m.

The 80% intervals from the state-space features contained the borehole depth at none of the 12 holes on those lines. They describe how well the inversion was copied, not how well the ground is known. The surrogate's mean error on the 11 uninverted-line holes of Table 2 (6.8–6.9 m) is lower than the smooth ensemble's (8.4 m), mostly because of two holes (1-G-96 and 2-G-96) where the ensemble finds conductive ground near the surface and the driller logged silt. What the ensemble adds is intervals that sometimes contain the truth.

## 4.6 Cost against MCMC

**The 2012 benchmark.** The trans-dimensional MCMC code of Minsley (2011), as configured in 2012, left a run log for a test sounding from an Alaskan survey, recorded on 11 July 2012. The settings were up to 30 layers between 2 and 150 m, a smoothness prior on the change in log resistivity between layers, and a relative data error solved for, with bird height and calibration held fixed. The chain counts as burned in at the first accepted model whose misfit reaches the target, and then runs 100,000 further iterations. The log records 99,681 burn-in iterations, so the sounding took about 200,000 forward evaluations and 1,425 s (about 7 ms per iteration).

**Table 4.** Cost per sounding, and for the survey, against the 2012 MCMC run.

| | MCMC (logged run, 2012) | Smooth ensemble | Sharp-contact ensemble |
|---|---|---|---|
| Forward evaluations per sounding | about 200,000 | 1,600 (400 × 4) | 1,600 (400 × 4) |
| Time per sounding | 1,425 s | 0.67 core-s | 0.14 core-s |
| All 71,470 soundings | about 3.2 years on one core | about 13 core-hours | about 3 core-hours |

The ensemble uses about 125 times fewer forward evaluations. In wall time the gap is about 2,100 times for the smooth model and 10,000 times for the sharp model, but part of that is the machine and language: the 2012 run was in MATLAB with live plotting enabled. A modern, parallel MCMC would close some of the wall-time gap, though not the factor of 125.

**The 2012 optimizations, measured.** In 2012 the author made two changes to the MCMC code. Neither was measured at the time; both are measured here, running the 2012 code in GNU Octave 8.4 with plotting off. Octave is slower than MATLAB, so the ratios between versions are the result, not the seconds.
1. **Vectorization of the code base.** Every `repmat` call was replaced by index-based tiling. Only one modified file survives (the reflection-coefficient routine), so the same change was applied to the four other files that call `repmat`. It gives identical numbers to the last digit. A single forward call with sensitivities is 34–51% faster, depending on the number of layers. With identical chains (same seed), 5,000 MCMC iterations ran 27% and 29% faster for two seeds. The reflection-coefficient routine alone accounts for 17–22%.
2. **Burn-in stopped on detection,** instead of a fixed 10,000 iterations. On eight soundings along line 30050, burn-in was detected after 411–8,325 iterations (median 2,700). That saves about 6% of a 110,000-iteration run. A fixed count is also unsafe: the Alaskan test sounding needed 37,453 iterations with seed 1, so a fixed 10,000 would have stopped it early.

Together, the changes cut the time per sounding by about 32%. The 2012 version also preallocated the chain once, where an earlier version grew it by one model per iteration. In Octave the two run equally fast, so the benefit of preallocation, which in MATLAB of that time could be large, could not be measured here.

# 5 Discussion

**Borrowing along the line is what makes the ensemble method work.** Run sounding by sounding from a broad prior, ES-MDA cannot fit the data: the median misfit stays at 2.0–2.3 whether it takes 4, 8 or 16 steps. The ensemble update is Gaussian. It can shift and stretch a cloud of models, but from a broad start the cloud spans models whose responses differ nonlinearly, and a linear step cannot reach the answer. Along the line, each sounding starts from its neighbour's posterior, so each update is small and nearly linear. The along-line prior is therefore not just a smoothness preference; it is what keeps the Kalman update in its linear regime. It also improves accuracy where it can be checked (Table 2: 1.7 m against 2.7 m on the inverted lines). The cost is that sharp lateral changes are smoothed at the scale of $L_h$, and the method needs flight lines dense enough for neighbours to resemble each other.

**What is inverted for matters more than how.** The smooth and sharp models use the same data, forward model, filter, smoother, gains and error model. They differ only in the question asked, and they place the valley floor about 10 m apart (section 4.3). A smooth Gaussian prior cannot represent a step, so a threshold on a smooth model fires early on the ramp above it. Asking for the depth of a contact directly removes that bias by construction. Trans-dimensional MCMC achieves something similar by letting the number and position of boundaries vary (Malinverno 2002; Minsley 2011). A fixed parameterization with an explicit contact gets much of that benefit at a small fraction of the cost, when the question is a single surface.

SHARP_DISCUSSION

**The intervals are conditional on the model.** A posterior describes uncertainty given its model: the parameterization, the one-dimensional earth, and the error model. When the model is close to right, as for the smooth model at shallow-bedrock holes, the intervals are honest (90% coverage). When it is rigid and sometimes wrong, as for the two-unit model where the stratigraphic contact is not the electrical one, they are far too narrow. The likelihood cannot detect this, because the error repeats from sounding to sounding and the along-line coupling then accumulates false confidence. The split-conformal step used here (Vovk et al. 2005; Angelopoulos & Bates 2023) is a cheap guard. It uses boreholes that are neither in the prior's validation set nor near enough to be used for validation, and it makes no assumption about why the model is wrong. Its guarantee rests on the calibration and validation holes being exchangeable. They are not quite: calibration holes lie 100–300 m from the nearest sounding, validation holes within 100 m, so the recalibrated intervals should err on the wide side.

**A calibration error in the survey data.** Two inversions of different design, one deterministic and one Bayesian, cannot fit the coaxial channels or the 8,171 Hz in-phase, and they miss them in the same direction by the same amount. The coaxial channels read about 12% high and the 8,171 Hz in-phase about 10% low, relative to the other channels. Survey-wide gains are a simple correction, but they cannot distinguish a calibration error from a systematic departure from the one-dimensional earth, and two of the gains (378 Hz in-phase, 129 kHz) are not identifiable from bird height and deep conductivity. The survey should be recalibrated, or its calibration records checked, before further quantitative use.

**Cost, and what it buys.** At 0.14–0.67 core-seconds per sounding, the whole survey can be inverted at every sounding in 3–13 core-hours. That makes it practical to re-invert as new boreholes arrive, to try alternative parameterizations, and to choose hyperparameters by marginal likelihood rather than by hand: the near-real-time goal of Friedel et al. (2015), reached by keeping the physics rather than replacing it with a learned surrogate. The comparison with MCMC is not like for like. MCMC samples a trans-dimensional posterior per sounding, while the ensemble approximates a fixed-dimensional one along the line, and its Gaussian update can understate non-Gaussian features such as the depth–resistivity trade-off of the sharp model. The synthetic test shows that trade-off clearly: with 100 members the ensemble committed to a contact twice as deep as the truth, under a more conductive fill, that fitted the data as well. With 400 members every interval contained the truth.

**Limitations.**
- The earth is assumed one-dimensional at each sounding. The buried valley is a two- or three-dimensional feature, and its walls may bias a 1D depth.
- Only every 20th sounding was used. At the full sampling rate, neighbouring data are near copies, and the along-line coupling would need a correlated error model.
- The relative data error was fixed, not estimated. A slowly varying error term along the line would let the likelihood see repeated errors.
- Validation rests on 21 holes, only two of them in the buried valley, and the depth prior comes from test holes that cluster where bedrock is shallow.

# 6 Conclusions

CONCLUSIONS

# Data and code availability

The code, tests, database build and learning notes are at https://github.com/aakbarie/Code-for-NZ. The raw survey and borehole files are listed, with checksums, in `manifests/raw_files.csv`. [Data-sharing terms for the survey and borehole data to be confirmed.]

# Acknowledgements

The author thanks Michael J. Friedel for the collaboration that framed this problem (Friedel et al. 2015), and Burke J. Minsley for the trans-dimensional MCMC code and system configuration used here as the benchmark (Minsley 2011). The survey was flown for the North Platte Natural Resources District; borehole records are from the Conservation and Survey Division, University of Nebraska–Lincoln.

# Use of AI tools

The analysis code (database, forward model, ensemble inversion, benchmarks) and a first draft of this manuscript were produced with Claude (Anthropic), an AI system, working under the author's direction. The author designed the study, set its scientific direction and validation criteria, supplied the domain knowledge and code history, and reviewed and takes responsibility for the content. AI tools do not meet authorship criteria and are not listed as authors. [Edit to match the target journal's policy.]

# References

Abraham JD, Cannia JC, Bedrosian PA, Johnson MR, Ball LB, Sibray SS (2012) Airborne electromagnetic mapping of the base of aquifer in areas of western Nebraska. U.S. Geological Survey Scientific Investigations Report 2011–5219, 38 p.

Angelopoulos AN, Bates S (2023) Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning 16:494–591. [verify]

Auken E, Christiansen AV, Jacobsen BH, Foged N, Sørensen KI (2005) Piecewise 1D laterally constrained inversion of resistivity data. Geophysical Prospecting 53:497–506. [verify author list]

Emerick AA, Reynolds AC (2013) Ensemble smoother with multiple data assimilation. Computers & Geosciences 55:3–15.

Evensen G (1994) Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. Journal of Geophysical Research 99(C5):10143–10162.

Evensen G (2009) Data assimilation: the ensemble Kalman filter, 2nd edn. Springer, Berlin.

Farquharson CG, Oldenburg DW, Routh PS (2003) Simultaneous 1D inversion of loop-loop electromagnetic data for magnetic susceptibility and electrical conductivity. Geophysics 68:1857–1869.

Friedman JH (2001) Greedy function approximation: a gradient boosting machine. Annals of Statistics 29:1189–1232.

Friedel MJ, Esfahani A, Iwashita F (2015) Toward real-time three-dimensional mapping of surficial aquifers using a hybrid modeling approach. Hydrogeology Journal. https://doi.org/10.1007/s10040-015-1318-2

Friedel MJ, Iwashita F (2013) Hybrid modeling of spatial continuity for application to numerical inverse problems. Environmental Modelling & Software 43:60–79.

Fugro Airborne Surveys (2008) RESOLVE survey for North Platte Resource District, North Platte River, Lodgepole Creek and Sidney Draw areas, western Nebraska. Report 08035, Fugro Airborne Surveys Corp., Mississauga, Ontario.

Iglesias MA, Law KJH, Stuart AM (2013) Ensemble Kalman methods for inverse problems. Inverse Problems 29:045001.

Kalman RE (1960) A new approach to linear filtering and prediction problems. Journal of Basic Engineering 82:35–45.

Malinverno A (2002) Parsimonious Bayesian Markov chain Monte Carlo inversion in a nonlinear geophysical problem. Geophysical Journal International 151:675–688.

Minsley BJ (2011) A trans-dimensional Bayesian Markov chain Monte Carlo algorithm for model assessment using frequency-domain electromagnetic data. Geophysical Journal International 187:252–272.

Minsley BJ, Foks NL, Bedrosian PA (2021) Quantifying model structural uncertainty using airborne electromagnetic data. Geophysical Journal International 224:590–607.

Raanes PN (2016) On the ensemble Rauch–Tung–Striebel smoother and its equivalence to the ensemble Kalman smoother. Quarterly Journal of the Royal Meteorological Society 142:1259–1264. [verify volume and pages]

Rasmussen CE, Williams CKI (2006) Gaussian processes for machine learning. MIT Press, Cambridge, MA.

Rauch HE, Tung F, Striebel CT (1965) Maximum likelihood estimates of linear dynamic systems. AIAA Journal 3:1445–1450.

Shumway RH, Stoffer DS (1982) An approach to time series smoothing and forecasting using the EM algorithm. Journal of Time Series Analysis 3:253–264.

Vovk V, Gammerman A, Shafer G (2005) Algorithmic learning in a random world. Springer, New York.

Ward SH, Hohmann GW (1988) Electromagnetic theory for geophysical applications. In: Nabighian MN (ed) Electromagnetic methods in applied geophysics, vol 1. Society of Exploration Geophysicists, Tulsa, pp 131–311.

Zhu K-G, Ma M-Y, Che H-W, Yang E-W, Yu S-B (2012) PC-based artificial neural network inversion for airborne time-domain electromagnetic data. Applied Geophysics 9:1–8. [verify]
