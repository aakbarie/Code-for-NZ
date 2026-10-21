# 02 · Spatial continuity, connectivity, and networks

A groundwater model needs a 3D picture of hydraulic conductivity (K). Two
different questions sit behind that picture:

- **Continuity:** how similar are values a given distance apart? Variograms
  answer this.
- **Connectivity:** do the permeable parts join up into pathways? Variograms
  can't answer this; connectivity and network measures can.

## Variograms

The variogram measures how different two values tend to be as a function of
the distance h between them:

```
γ(h) = ½ · average of (Z(x + h) − Z(x))²   over all pairs about h apart
```

It usually rises with distance and levels off:

- the **nugget** is the jump at h ≈ 0 (measurement error plus variation finer
  than the sampling);
- the **sill** is the level it flattens at (the total variance);
- the **range** is the distance where it flattens (beyond it, values are
  unrelated).

If it rises faster in one direction than another, the field is anisotropic.

Three uses matter for groundwater:

1. **Kriging:** the best linear estimate at unsampled points, given the
   variogram. It is smooth by design, so it understates extremes.
2. **Sequential Gaussian simulation (SGS):** many equally likely maps that
   honour the data and the variogram. The spread across maps is the
   uncertainty.
3. **Pilot points (PEST):** when calibrating a model such as MODFLOW, K is
   adjusted at a set of pilot points and interpolated between them. The
   variogram ties neighbouring pilot points together, so the calibration
   can't produce geologically silly patterns (Doherty, 2003).

## The sparse-data problem, and the 2013 idea

A reliable variogram needs on the order of 100–150 well-spread points (Webster
& Oliver, 1992). Morrill has 32 logged holes and 130 with lithology, clustered
along a few transects.

Friedel & Iwashita (2013) proposed a way around this:

1. Train a self-organizing map on the sparse measurements together with dense,
   cheap covariates (terrain, airborne EM).
2. Use it to estimate the property at thousands of random locations.
3. Fit variograms to those estimates.
4. Simulate with SGS.
5. Hand the results to the inverse model as starting values and constraints.

The idea is sound and was ahead of its time: let abundant secondary data
inform spatial structure, and carry machine learning into the inverse problem.

There is one statistical catch worth understanding. A variogram fitted to
*estimates* describes the estimator, not the property. The law of total
variance splits the variance of a property Z, given covariates X, into two
parts:

```
Var(Z) = Var( E[Z | X] )  +  E[ Var(Z | X) ]
          explained by the      what the covariates
          covariates             can't explain
```

An estimate carries only the first part. So the variogram of estimates has:

- **too low a sill**, because the unexplained variance is missing;
- **too small a nugget**;
- **a range borrowed from the covariates**, e.g. the scale of the terrain.

Simulations built on it come out too smooth and too confident.

**How this project will do it:**

1. **Split the property into trend and residual.** Model Z = m(X) + R, where
   m is learned from covariates by machine learning and checked with spatial
   cross-validation.
2. **Fit the variogram to the residuals.** Compute R at the boreholes and fit
   its variogram. There are few points, but they only need to describe what
   m leaves over. This is regression kriging.
3. **Simulate R and add the trend back.** Use a model that predicts a
   distribution rather than one number (e.g. a quantile regression forest),
   so its uncertainty carries through as well.
4. **Test it on a synthetic aquifer with a known answer.** Sample it the way
   Morrill is sampled. Calibrate MODFLOW 6 with PEST++ pilot points three
   ways: no prior, the 2013-style variogram, and the residual approach.
   Compare errors and how honest the uncertainty bands are.

## Connectivity: what variograms can't see

A variogram compares values two points at a time. Two fields can share the
same histogram and the same variogram yet move water completely differently:
gravel connected in long channels versus gravel in isolated lenses (Zinn &
Harvey, 2003).

Morrill has exactly such a feature. Along easting ≈ 588 km, between
northings 4,643.7 and 4,648 km, the CSD holes (1-G-96, 1-I-96, 1-H-97,
1-J-96, 2-J-97, 2-L-96, 1-L-96) show 13–60 m of sand and gravel. Bedrock there
is Chadron or even Pierre Shale, down to about 60 m. Just to the south, most
K-94 holes hit Brule siltstone within 1–7 m. My interpretation, to be tested
with the airborne data, is that this is a buried, gravel-filled valley. If it
is continuous, it is a fast pathway for water and anything dissolved in it,
and no variogram would tell you so.

Connectivity measures (Renard & Allard, 2013) include:

- the share of permeable cells in the largest connected cluster;
- whether a cluster spans the domain (percolation);
- the connectivity function τ(h): the probability that two permeable points h
  apart are joined through permeable cells.

## Networks and small worlds

Watts & Strogatz (1998) described networks that sit between a regular lattice
and a random graph:

- **high clustering** C, like a lattice: neighbours of a node are neighbours
  of each other;
- **short paths** L, like a random graph: any node is a few steps from any
  other.

A few long-range "shortcuts" are enough to collapse distances. Two common
scores, each comparing against equivalent random (rand) and lattice (latt)
networks:

- σ = (C / C_rand) / (L / L_rand): small-world if clearly above 1
  (Humphries & Gurney, 2008).
- ω = L_rand / L − C / C_latt: ranges from −1 (lattice-like) to +1
  (random-like); near 0 means small-world (Telesford et al., 2011).

There are two ways to turn this project's data into a network.

**A · Flow network (physical).**
- **Nodes:** cells of the 3D aquifer model.
- **Edges:** between neighbouring cells.
- **Weights:** travel time, t = distance × porosity / (K × gradient), using the
  harmonic mean of K for flow from one cell into the next.

In travel-time terms a gravel channel is a shortcut. One caveat: counting
*hops* on a plain grid can never look small-world. Paths on a grid are long
by construction, and a 6-neighbour cubic grid has no triangles at all
(C = 0). The small-world question only makes sense with travel-time
weights.

What it gives:
- fastest flow paths;
- **betweenness centrality**: cells that many fast paths squeeze through,
  which are the chokepoints for contamination or pumping;
- whether the gravel connects recharge areas to the river.

It can be checked against particle tracking in MODFLOW/MODPATH.

**B · Similarity network (statistical).**
- **Nodes:** airborne soundings.
- **Edges:** link two soundings whose resistivity–depth profiles are very
  similar, however far apart they are.

This is how "climate networks" are built from grid points with correlated
records (Tsonis & Roebber, 2004; Donges et al., 2009). Nearby soundings link
because of continuity, which gives clustering. Distant soundings link when the
same deposit repeats, e.g. separate stretches of the same buried valley; those
are the shortcuts. Communities found in the network (Louvain/Leiden) are
candidate hydrostratigraphic units, an alternative to the 2015 SOM clusters
that can be checked against the borehole lithology.

Pitfalls to avoid:
- **Spatial autocorrelation inflates clustering.** Compare against
  spatially autocorrelated null models (e.g. randomly shifted or rotated
  fields), not only random graphs.
- **The threshold drives the result.** Scan the similarity threshold rather
  than picking one.

**Order:** B needs only the inverted resistivities, so it can start as soon as
`inv_model_pt1.csv` is loaded. A needs the 3D K model from Steps 2–3.

```
model_resistivity (sounding × layer)            lithology, aquifer_base (boreholes)
        │                                                   │
        ├─ B: similarity network                            │
        │     nodes = soundings, edges = similar profiles   │
        │     ─► C, L, σ, ω against spatial null models     │
        │     ─► communities = candidate units ─────────────┤ check
        │                                                   │
        └─ Step 2 model ─► 3D K grid ─► A: flow network     │
              nodes = cells, edges = neighbours, weight = travel time
              ─► connected clusters, fastest paths, betweenness
              ─► compare with MODFLOW 6 / MODPATH particle paths
```

Tools: `networkx` for prototypes; `igraph` (or `graph-tool`) and
`scipy.sparse.csgraph` for graphs with 10⁵–10⁶ nodes.

## Try it

```sql
-- Depth to bedrock at each hole (run `uv run morrill build` first)
SELECT hole_id, round(x) AS x, round(y) AS y, round(bedrock_depth_m, 1) AS bedrock_m, bedrock_unit
FROM aquifer_base WHERE reached_bedrock ORDER BY bedrock_depth_m DESC;
```

## Questions to think about

1. Why can't a variogram distinguish one connected gravel channel from many
   scattered gravel lenses with the same total volume?
2. In network A, why use the *harmonic* mean of K between two cells, and what
   happens to travel time if one of them is clay?
3. In network B, if you lower the similarity threshold until everything links,
   what happens to C and L? Where would you expect the interesting regime?

## References

- Doherty, J. (2003). Ground water model calibration using pilot points and regularization. *Ground Water* 41(2), 170–177.
- Donges, J. F., et al. (2009). Complex networks in climate dynamics. *Eur. Phys. J. Special Topics* 174, 157–179.
- Friedel, M. J., & Iwashita, F. (2013). Hybrid modeling of spatial continuity for application to numerical inverse problems. *Environmental Modelling & Software* 43, 60–79.
- Humphries, M. D., & Gurney, K. (2008). Network "small-world-ness". *PLoS ONE* 3(4), e0002051.
- Renard, P., & Allard, D. (2013). Connectivity metrics for subsurface flow and transport. *Advances in Water Resources* 51, 168–196.
- Telesford, Q. K., et al. (2011). The ubiquity of small-world networks. *Brain Connectivity* 1(5), 367–375.
- Tsonis, A. A., & Roebber, P. J. (2004). The architecture of the climate network. *Physica A* 333, 497–504.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of "small-world" networks. *Nature* 393, 440–442.
- Webster, R., & Oliver, M. A. (1992). Sample adequately to estimate variograms of soil properties. *Journal of Soil Science* 43, 177–192.
- Zinn, B., & Harvey, C. F. (2003). When good statistical models of aquifer heterogeneity go bad. *Water Resources Research* 39(3), 1051.
