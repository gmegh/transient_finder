# Dark cosmic roundness workflow notes

## Butler queries for generated products

Replace the placeholders with your concrete data IDs.

```python
from lsst.daf.butler import Butler

butler = Butler("/repo/main", collections=["u/gmegias/transient_catalogs", "LSSTCam/defaults"])

# Per-detector source catalog from detection task
cat = butler.get(
    "detector_dark_source_catalog",
    instrument="LSSTCam",
    exposure=123456,
    detector=42,
)

# Per-detector subset: round + fat sources used in the footprint montage
round_fat_cat = butler.get(
    "detector_round_fat_dark_source_catalog",
    instrument="LSSTCam",
    exposure=123456,
    detector=42,
)

# Plot object (matplotlib Figure storage class)
fig = butler.get(
    "detector_round_fat_dark_source_plot",
    instrument="LSSTCam",
    exposure=123456,
    detector=42,
)
fig.savefig("round_fat_dark_footprints.png", dpi=200, bbox_inches="tight")
```

## One-image test suggestion

Use your single-exposure BPS submit YAML and change `payload.dataQuery` to one known
embargoed exposure in your data repository.
