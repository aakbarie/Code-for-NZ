# 00 · The big picture

These notes explain *why* each part of this project is built the way it is.
They are written for studying alongside the code. Each note ends with a few
questions and queries to try.

## From a helicopter to a groundwater model

```
helicopter EM survey ──► resistivity of the ground ──► rock type ──► how easily water moves ──► groundwater model
   (what we measure)        (Step 2: ML + stats)        (Step 3)        (hydraulic conductivity)      (MODFLOW)
                                    ▲
                     boreholes: the ground truth that checks every step
```

1. **Measure.** A helicopter tows a sensor ("bird") about 30–40 m above the
   ground. A transmitter coil drives an alternating magnetic field at six
   frequencies (about 400 Hz to 140 kHz). That field induces electric currents in
   the ground; the currents make their own weak magnetic field, which a receiver
   coil picks up. Low frequencies reach deeper, high frequencies see the near
   surface. The result: 12 numbers per sounding (in-phase and quadrature at six
   frequencies), recorded every few metres along flight lines.

2. **Resistivity.** Those 12 numbers depend on how electrically resistive the
   ground is at each depth. Recovering resistivity versus depth is called
   *inversion*. The usual way fits a layered model to each sounding; Burke
   Minsley's MCMC method does it thousands of times per sounding to get
   uncertainty, which is slow. This project's goal is to learn the mapping with
   machine learning and statistics instead, and check it against boreholes.

3. **Rock type.** Resistivity is a proxy. Clay conducts electricity well (low
   resistivity, roughly 1–20 ohm·m); clean sand and gravel saturated with fresh
   water conduct poorly (high resistivity, often 50–1,000+ ohm·m). Boreholes with
   driller's logs let us learn which resistivities go with which materials *here*.

4. **Hydraulic conductivity (K).** How easily water flows through a material.
   It spans about twelve orders of magnitude between clay and gravel, which is
   why getting the rock type right matters so much more than small resistivity
   errors.

5. **Groundwater model.** A grid of cells, each with a K value, plus recharge,
   rivers and wells. MODFLOW solves for water levels and flow. The 3D aquifer map
   from steps 2–4 becomes its geology.

## The data, in one table each

| Data | What it tells us | Where in the database |
|---|---|---|
| Airborne EM soundings | The 12-number response at ~70,000 places | `sounding`, `hem_obs` |
| Inversions (EM1DFM, MCMC, 2015 SOM) | Resistivity in 24 layers at each sounding | `model_run`, `model_resistivity` |
| Borehole induction logs | Resistivity every 3 cm down a hole | `induction_log` |
| Driller's lithology | Rock type by depth interval | `lithology`, `lithology_class` |

## Notes in this folder

- [01 · Boreholes](01-boreholes.md): induction logs, lithology, and how they become ground truth.
- More are added as each step is built.
