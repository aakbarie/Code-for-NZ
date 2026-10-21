# 2015 R scripts (archived)

These two scripts prepared self-organizing map (SOM) output for 3D display in
Golden Software Voxler, for the 3D resistivity blocks in Fig. 8 of
Friedel, Esfahani & Iwashita (2015). They are kept here unchanged as a record
of what was done for the paper. Nothing in the new Python project uses them.

| File | Date | What it did |
|---|---|---|
| `wide_to_long_original.R` | 2015-05-13 | Reshaped `Morrill_block_test_*_voxler.csv` from one row per sounding (`res1`…`res24`) to one row per sounding × layer, and attached layer numbers from `layer_depth.csv` |
| `wide_to_long_for_nz.R` | 2015-05-21 | Same, plus layer thickness and cumulative thickness, and computed `DEM − cumThick` |

Input files (not in git) are in Google Drive under
`My Drive/Mikes/Code-for-NZ/Wide-to-Long-NZ/`.

## Known problems

- Hard-coded Windows paths (`C:\Users\...\Google Drive\Mikes\...`) and `setwd()`,
  so they only ran on the original laptop.
- The `depth` column ends up holding the **elevation** of each layer's bottom
  (`DEM − cumThick`), not a depth.
- `wide_to_long_for_nz.R` loops over every row of `layer_depth.csv` but indexes
  a 21-row subset, so rows 22–24 are `NA`. It still works only because R
  ignores `NA` indexes when assigning a single value.
- `wide_to_long_original.R` calls `paste()` without `sep = ""`, which produced
  file names with a space (`...voxler.csv _short.csv`). Both scripts produce
  double extensions (`X.csv_long.csv`).
- "Short" output keeps layers 1–21 and "long" keeps 1–24. Layers 22–24 have
  essentially no real inverted values (one each; paper Table 3), so their values
  in the SOM output are imputed.
- An output file produced by an intermediate version was committed in May 2015
  and later deleted. It had the elevation filled in for only 24 of its
  1,127,616 rows. That file has since been purged from git history.
