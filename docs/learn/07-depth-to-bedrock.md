# 07 · Inverting for the depth to bedrock directly

Note 06 found the shale under the buried valley at about the right depth, but
put the floor of the valley 10–25 m too high. The reason was the question
being asked, not the method:
- **The model:** smooth resistivity in 25 fixed layers, so it ramps down to
  the shale over 10–20 m.
- **The rule:** the first layer under 20 ohm·m counts as the conductor, so
  it fires near the top of the ramp.

This note changes the question. The depth to bedrock becomes a number the
inversion solves for, with a sharp contact at that depth.

## The model

At each sounding:

```
θ = [log10 ρ of 5 fill layers,  log10 z_b,  log10 ρ of the bedrock,  δh]
```

- **The fill** runs from the surface to z_b, split into five layers in fixed
  proportions of z_b: 5, 10, 15, 30 and 40%. Every member of the ensemble
  has the same structure but its own depth.
- **Below z_b** is a bedrock half-space with one resistivity: the sharp
  contact.
- **δh** corrects the bird height, as before.

That is 8 numbers instead of 26. Each forward call also has 6 layers instead
of 25, so it is cheaper. The forward model now accepts a different set of
thicknesses for every member (`fdem.response`, tested against one call per
model).

## The prior, from boreholes the checks don't use

As in note 06, the prior comes only from boreholes with no sounding within
100 m, so the validation holes stay unseen:
- **Depth to bedrock:** 78 holes; median 6.1 m, spread 0.43 decades (a
  factor of 2.7).
- **Fill:** about 39 ohm·m, from induction logs above the contact (8 holes).
- **Bedrock:** about 14 ohm·m, from logs below the contact.

The depth prior is biased shallow. Test holes cluster where bedrock is
shallow and cheap to reach: 65 of the 78 bottom in the Brule Formation within
a few metres. A 60 m valley floor sits about 2.3 spreads above this median.
How far to widen it is chosen by marginal likelihood, along with the
along-line length and the widening of the resistivity spreads.

Everything else is note 06's machinery, unchanged:
- the along-line AR(1) prior;
- ES-MDA at each sounding;
- the ensemble RTS smoother;
- the channel gains;
- the 5% data error.

## Why the ensemble needs 400 members here

The synthetic test puts 150 ohm·m gravel on 8 ohm·m shale at 45 m, with a
prior median of 6 m. It taught something:
- **With 400 members,** two noise draws give medians of 42–55 m, and every
  80% interval contains 45 m.
- **With 100 members,** one draw sends the whole ensemble to a contact at
  115–140 m that fits the data equally well.

Depth and fill resistivity trade off: a deeper contact under a more
conductive fill gives almost the same response. A small ensemble can commit
to the wrong side of that trade-off and never come back. The settings search
therefore uses 200 members, and the final run 400.

## Results

RESULTS

## Questions to think about

1. The depth prior comes from test holes that cluster where bedrock is
   shallow. What would an unbiased prior look like, and how could you build
   one from data you have?
2. With 100 members the ensemble found a deep contact under a conductive
   fill that fit the data as well as the truth. What extra information would
   break that tie?
3. The sharp model has 8 unknowns, the smooth model 26. Which do you expect
   to have the better-calibrated intervals, and why?
