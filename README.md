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
| 1 | A single, documented database of the survey, inversions, and borehole data | in progress |
| 2 | Resistivity from ML + statistics, validated by holding out whole boreholes, with calibrated uncertainty; benchmarked against the 2015 SOM results and MCMC | planned |
| 3 | From resistivity to lithology, hydraulic properties, and a groundwater model | planned |

## Data

Raw data is **not** stored in git. It lives in Google Drive, and
[`manifests/raw_files.csv`](manifests/raw_files.csv) lists every input file
with its Drive location and SHA-256 checksum. Download the files into
`data/raw/` (keeping the paths in the manifest), then check them:

```bash
uv run morrill verify-raw
```

## Layout

```
src/morrill/      Python package (database build, loaders, checks)
manifests/        list of raw input files with checksums
tests/            automated tests
docs/learn/       plain-language notes explaining each step
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
