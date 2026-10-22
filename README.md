# Morrill block: airborne EM and boreholes to aquifer maps

This project rebuilds, with modern tools, the work behind

> Friedel, M. J., Esfahani, A., & Iwashita, F. (2015). Toward real-time
> three-dimensional mapping of surficial aquifers using a hybrid modeling
> approach. *Hydrogeology Journal*. https://doi.org/10.1007/s10040-015-1318-2

The study area is the Morrill block in the North Platte River valley, Scotts
Bluff County, Nebraska. A helicopter frequency-domain EM survey (Fugro RESOLVE,
June 2008) measured the ground's response at six frequencies every few metres
along flight lines. Turning those measurements into subsurface resistivity is
usually done by inversion, one sounding at a time; a Bayesian MCMC inversion
(Minsley, 2011) also gives uncertainty but is slow. The aim here is to get
resistivity, with honest uncertainty, from machine learning and statistics
trained against inversions and checked against borehole logs, in a fraction of
the time.

## Plan

| Step | What | Status |
|---|---|---|
| 1 | A single, documented database of the survey, inversions, and borehole data | done for the survey, the EM1DFM inversion and all borehole data; other inversions and the 2015 results still to load |
| 2 | Resistivity from ML + statistics, validated by holding out whole boreholes, with calibrated uncertainty; benchmarked against the 2015 SOM results and MCMC | planned |
| 3 | From resistivity to lithology, hydraulic conductivity, and a MODFLOW 6 groundwater model. Builds on the spatial-continuity approach of Friedel & Iwashita (2013): ML trend plus geostatistics on the residuals, tested on a synthetic aquifer with PEST++ pilot points | first pieces: histogram (Bayes) model of rock class from resistivity ([note 04](docs/learn/04-histogram-lithology-model.md)); volume of aquifer material from the bedrock ensemble ([note 07](docs/learn/07-depth-to-bedrock.md)); a flow model needs water levels, aquifer tests, wells, river and canal data |
| + | Network analysis: a similarity network of soundings (small-world structure, hydrostratigraphic communities) and a travel-time flow network of the 3D aquifer (connectivity, preferential paths) | first result: UMAP + Leiden communities ([note 03](docs/learn/03-airborne-and-communities.md)) |
| + | Kalman state-space features along flight lines: flight height removed inside the model, along-line smoothing with uncertainty, surprise scores | first result ([note 05](docs/learn/05-kalman-features.md)) |
| 2 | Ensemble Kalman inversion: our own 1D EM forward model plus an ensemble Kalman filter and smoother along each line, with a prior learned from the borehole logs; every line inverted with uncertainty | done ([note 06](docs/learn/06-ensemble-kalman-inversion.md), [note 07](docs/learn/07-depth-to-bedrock.md)): all 12 lines in 2–10 min; beats EM1DFM at shallow-bedrock holes; inverting for the depth to bedrock cuts the buried-valley error; composing a depth grid exactly along each line ([note 08](docs/learn/08-profunctor-inversion.md)) gives 3.7 m mean error at 21 held-out holes; intervals recalibrated on held-out boreholes |

See [docs/learn/02](docs/learn/02-continuity-connectivity-networks.md) for the
reasoning behind Steps 3 and +.

## Data

Raw data is **not** stored in git. It lives in Google Drive, and
[`manifests/raw_files.csv`](manifests/raw_files.csv) lists every input file
with its Drive location and checksum.
[docs/data-sources.md](docs/data-sources.md) explains what each file is,
coordinate systems and units, known data problems, and what is still missing.

Download the files into `data/raw/` (keeping the paths in the manifest), then:

```bash
uv run morrill verify-raw   # which files are present, missing, or the wrong version
uv run morrill build        # build data/morrill.duckdb and run the data checks

uv sync --extra analysis    # once: UMAP, Leiden, matplotlib
uv run morrill communities  # group soundings by resistivity profile, write figures
uv run morrill lithology    # P(coarse / fine / bedrock) in every model cell, with validation
uv run morrill kalman       # Kalman state-space features along every flight line (~12 min)
uv run morrill enki         # ensemble Kalman inversion of every line, with checks (~25 min)
uv run morrill bedrock      # depth to bedrock as a sharp contact, with calibrated intervals (~5 min)
uv run morrill profunctor   # depth to bedrock on a grid, composed exactly along each line (~7 min)
uv run morrill aquifer      # thickness and volume of aquifer material, by inversion (seconds)
```

## Layout

```
src/morrill/      Python package (database build, loaders, checks, analysis/)
manifests/        list of raw input files with checksums
tests/            automated tests
docs/learn/       plain-language notes explaining each step
docs/figures/     figures used in the notes
legacy/2015-r/    the original 2015 R scripts, kept for the record
data/             raw files and the built database (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## License

Artistic License 2.0. See [LICENSE](LICENSE).
