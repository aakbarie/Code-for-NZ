# 01 · Boreholes: the ground truth

The airborne survey covers the whole block but measures the ground only
indirectly. Boreholes are the opposite: direct, but at a handful of points.
Every model in this project is judged by how well it agrees with them, so it is
worth knowing exactly what they contain and where they are weak.

## What we have

The holes are test holes drilled by the Nebraska Conservation and Survey
Division (CSD). Names look like `1-F-96`: hole 1 of series F, drilled in 1996.

| | Holes | Rows | Source file |
|---|---|---|---|
| Induction logs | 32 | 37,394 readings | `CSD_ires.csv` |
| Driller's lithology | 130 | 1,173 intervals | `CSD_Lith.csv` |
| Formation picks | 102 | 210 intervals | `CSD_strat.csv` |

All 32 logged holes also have lithology; the other 98 have lithology only.

## Induction logs

An induction tool is lowered down the hole. Like the helicopter system, it uses
a coil to induce currents in the surrounding rock and measures the result. It
reports **electrical conductivity** in millisiemens per metre (mS/m), every
0.1 ft (3.048 cm). Resistivity is the reciprocal:

```
resistivity (ohm·m) = 1000 / conductivity (mS/m)
```

Things to know about these logs:

- **They start about 1 m down.** The top metre is casing or disturbed ground,
  so the thinnest top model layer (0–0.86 m) has no log data at all.
- **Very low readings are noise.** Below about 1 mS/m (above 1,000 ohm·m) the
  tool is near its noise floor; a reading of 0.004 mS/m would imply
  250,000 ohm·m, which is not physical here. 98 readings fall below 1 mS/m and
  are flagged in the `induction_log_clean` view. The 2015 paper also removed
  values above 1,000 ohm·m.
- **The delivered resistivity column disagrees in about 2% of rows.** The file
  has both `COND` and `RES_`, and `RES_` isn't always `1000/COND`; the
  differences sit at sharp boundaries and near-zero conductivities. Since the
  tool measures conductivity, the database treats `COND` as the measurement and
  keeps `RES_` only for reference.
- **Few holes are deep.** Layer by layer, this is how many holes have log data:

  | Model layers | Depth (m) | Holes with log data |
  |---|---|---|
  | 2–8 | 0.9–11 | 32 |
  | 13 | 23–27 | 23 |
  | 16 | 36–42 | 11 |
  | 18 | 48–55 | 8 |
  | 19 | 55–64 | 4 |
  | 20–24 | 64–125 | 1 (hole 14-K-94, logged to 154 m) |

  Below about 50 m there is almost no ground truth. Any claim a model makes
  about the deep layers can barely be checked, and that limit applies to the
  2015 paper as much as to this project.

## Driller's lithology

Drillers describe what comes up the hole. The file has 39 distinct labels, cut
off at 16 characters ("Silt to siltston"). The database keeps the label as
delivered and maps it, through
[`src/morrill/reference/lithology_classes.csv`](../../src/morrill/reference/lithology_classes.csv),
to three things a hydrogeologist cares about:

- **texture**: clay, silt, silty sand, sand, sand and gravel, gravel, mudrock
  (siltstone, claystone, mudstone, shale), sandstone, ash, soil, fill.
- **consolidated**: loose sediment or bedrock.
- **aquifer role**: aquifer or confining unit, following the paper's binary
  split (sand, gravel, sandstone vs. clay, silt, siltstone, shale).

What the 130 holes contain, by total thickness logged:

| Texture | Role | Thickness (m) |
|---|---|---|
| mudrock | confining | 975 |
| sand | aquifer | 745 |
| sand and gravel | aquifer | 556 |
| silt | confining | 492 |
| clay | confining | 481 |
| gravel | aquifer | 369 |
| sandstone | aquifer | 130 |
| soil, fill, ash, other | — | ~70 |

**The mapping is a judgment call. Please review it.** Examples:

- "Sand and silt" is treated as an aquifer, but "Silt and sand" as confining,
  on the convention that the first-named material dominates.
- **Brule** is the Brule Formation, a White River Group siltstone. It is mapped
  as confining, but fractured Brule yields water locally in western Nebraska.
  That is a real hydrogeologic ambiguity the model will have to live with.

### Why texture matters for flow

Hydraulic conductivity (K) measures how easily water moves through a material.
Rough ranges, after Freeze & Cherry (1979), in metres per second:

| Material | K (m/s), order of magnitude |
|---|---|
| Gravel | 10⁻³ to 1 |
| Clean sand | 10⁻⁶ to 10⁻² |
| Silty sand | 10⁻⁷ to 10⁻³ |
| Silt | 10⁻⁹ to 10⁻⁵ |
| Clay | 10⁻¹² to 10⁻⁹ |
| Sandstone | 10⁻¹⁰ to 10⁻⁶ |
| Shale, siltstone | 10⁻¹³ to 10⁻⁹ |

Gravel can transmit water a billion times faster than clay. So in Step 3, telling
sand from silt matters far more than knowing whether a layer is 40 or 45 ohm·m.

### Why resistivity is only a proxy for texture

In clean sand or gravel, electricity flows mostly through the pore water.
Archie's law captures this: bulk resistivity ≈ pore-water resistivity ×
porosity^(−m), with m around 1.5–2.5. Clay adds a second path along the
mineral surfaces, which lowers resistivity regardless of the water. So:

- clay → low resistivity (roughly 1–20 ohm·m)
- sand and gravel with **fresh** water → high (often 50–1,000+ ohm·m)
- sand with **salty** water → low as well

That last case is why the paper wanted pore-water specific conductivity: without
it, salty sand and clay can look alike. The raw water-quality data hasn't been
found (see [data sources](../data-sources.md)), so for now this ambiguity remains.

### What the logs actually show in this valley

Joining every reliable log reading to the lithology interval it falls in gives
the median resistivity of each texture. Taking the median per hole first, so
that one long hole can't dominate:

| Texture | Holes | Median of hole medians (ohm·m) | Range across holes |
|---|---|---|---|
| mudrock (siltstone, Brule, …) | 14 | 12 | 8–104 |
| silt | 4 | 27 | 21–48 |
| gravel | 10 | 35 | 1–60 |
| clay | 5 | 39 | 22–72 |
| sand and gravel | 15 | 41 | 30–83 |
| sand | 25 | 45 | 24–143 |

Two lessons:

1. **The strongest contrast is loose sediment against bedrock.** Consolidated
   mudrock is about 3× more conductive than the overlying sand and gravel. That
   boundary is the **base of the aquifer**, and it is what airborne EM maps best
   here. The USGS used airborne EM in western Nebraska largely for that purpose
   (Abraham et al., 2012, USGS SIR 2011-5219), and the 2015 paper used their
   base-of-aquifer surface to trim its data.
2. **Within the sediments, textures overlap.** Between 5 and 15 m depth the
   textbook order does appear (clay 28 < sand and gravel 44 < sand 53 <
   gravel 62 ohm·m), but the spread inside each class is as large as the gaps
   between them.

One hole distorts the clay numbers: **1N-97** has 18 m of "Clay" (7.6–25.9 m)
that logs at about 71 ohm·m, far above any other clay. Either the driller's
label or the depth registration there is suspect, or this clay is unusual
(dry, sandy, or cemented). It is worth checking against the original CSD record
before using that hole for training.

## From a log to a model layer

To compare a log with the airborne layers, the 3 cm readings have to be
averaged over each layer's thickness. The airborne coils are horizontal and
drive currents sideways through the layers, so the layers act like resistors
in **parallel**, where conductances add. The right average is therefore the
mean of **conductivity**:

```
layer resistivity = 1000 / mean(conductivity of the readings in the layer)
```

Averaging resistivity directly gives too much weight to thin resistive beds.
Example: two equal readings of 10 and 1,000 mS/m (100 and 1 ohm·m) average to
1.98 ohm·m in conductivity, but 50.5 ohm·m if you average the resistivities.

The `borehole_layer_resistivity` view computes both, so you can see the
difference in the real data. **It turned out smaller than I expected.** The
median difference is about 1%. In a handful of hole-layers it reaches 2–8×,
where a thin resistive streak sits in a conductive layer. So averaging alone
doesn't explain the weak 2015 log-vs-airborne correlation (ρ = 0.28 for
layer 6). The scale mismatch (a 3 cm reading vs. an airborne footprint tens
of metres wide) is the more likely cause.

The view also reports **coverage**: the share of the layer that the log
actually samples. Layer 2 averages only 0.65 because logs start around 1 m.
Filter on coverage before trusting a layer value.

## Try it

```sql
-- Build first (uv run morrill build), then open the database:
--   uv run python -c "import duckdb; duckdb.connect('data/morrill.duckdb').sql('...').show()"

-- Holes with both logs and lithology
SELECT hole_id, round(collar_elev_m, 1) AS collar FROM borehole
WHERE has_induction_log AND has_lithology ORDER BY hole_id;

-- One hole, layer by layer: conductivity mean vs naive mean
SELECT layer, coverage, round(resistivity_ohm_m, 1), round(resistivity_naive_ohm_m, 1)
FROM borehole_layer_resistivity WHERE hole_id = '14-K-94' ORDER BY layer;

-- Typical log resistivity for each texture
SELECT c.texture, round(median(l.resistivity_ohm_m), 1) AS median_ohm_m, count(*) AS readings
FROM induction_log_clean AS l
JOIN lithology_classified AS c
  ON l.hole_id = c.hole_id AND l.depth_m >= c.from_m AND l.depth_m < c.to_m
WHERE l.reliable
GROUP BY c.texture ORDER BY median_ohm_m;
```

## Questions to think about

1. Sand, gravel and clay overlap in resistivity here, but mudrock stands apart.
   What does that mean for predicting aquifer vs. confining unit from
   resistivity alone? Which hydrologic question can resistivity answer well,
   and which can't it?
2. Below 50 m only one hole has a log. How would you validate a model's deep
   layers honestly? What should a model report there instead of a single number?
3. Brule siltstone is mapped as confining. Where in the data could you look for
   signs that it behaves like an aquifer in some places?
