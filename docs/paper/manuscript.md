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
directly. The price is computation. A single sounding in the North Platte
River valley took about 200,000 forward-model evaluations and 24 minutes in
2012 (section 4.6). A survey of 70,000 soundings would need years of
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

RESULTS

# 5 Discussion

DISCUSSION

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

Auken E, Christiansen AV, Jacobsen BH, Foged N, Sørensen KI (2005) Piecewise 1D laterally constrained inversion of resistivity data. Geophysical Prospecting 53:497–506. [verify author list]

Emerick AA, Reynolds AC (2013) Ensemble smoother with multiple data assimilation. Computers & Geosciences 55:3–15.

Evensen G (1994) Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. Journal of Geophysical Research 99(C5):10143–10162.

Evensen G (2009) Data assimilation: the ensemble Kalman filter, 2nd edn. Springer, Berlin.

Farquharson CG, Oldenburg DW, Routh PS (2003) Simultaneous 1D inversion of loop-loop electromagnetic data for magnetic susceptibility and electrical conductivity. Geophysics 68:1857–1869.

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

Ward SH, Hohmann GW (1988) Electromagnetic theory for geophysical applications. In: Nabighian MN (ed) Electromagnetic methods in applied geophysics, vol 1. Society of Exploration Geophysicists, Tulsa, pp 131–311.

Zhu K-G, Ma M-Y, Che H-W, Yang E-W, Yu S-B (2012) PC-based artificial neural network inversion for airborne time-domain electromagnetic data. Applied Geophysics 9:1–8. [verify]
