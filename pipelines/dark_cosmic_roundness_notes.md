# Dark cosmic roundness workflow notes

## What is now produced

The per-exposure export task keeps only sources passing both:

- `roundness >= minRoundness`
- `equivalent_diameter >= minDiameterPixels` (default set to 6 px)

and then saves only up to `nSources` cutouts (default 25).

## Browseable cutout folder

The task writes PNG files under:

`artifacts/round_fat_dark_cutouts/<instrument>/exposure_<exposure>/`

Each file is named with detector, source id, and roundness.

## Butler queries for generated products

Replace placeholders with concrete data IDs.

```python
from lsst.daf.butler import Butler

butler = Butler("/repo/main", collections=["u/gmegias/transient_catalogs", "LSSTCam/defaults"])

# Per-exposure consolidated catalog (all detectors combined)
all_cat = butler.get(
    "exposure_dark_source_catalog",
    instrument="LSSTCam",
    exposure=123456,
)

# Per-exposure round+fat subset actually exported as cutouts
round_fat_cat = butler.get(
    "exposure_round_fat_dark_source_catalog",
    instrument="LSSTCam",
    exposure=123456,
)

# Per-exposure montage figure
fig = butler.get(
    "exposure_round_fat_dark_source_plot",
    instrument="LSSTCam",
    exposure=123456,
)
fig.savefig("round_fat_dark_footprints_exposure123456.png", dpi=200, bbox_inches="tight")
```

## One-image test suggestion

Use your single-exposure BPS submit YAML and change `payload.dataQuery` to one known
embargoed exposure in your data repository.
