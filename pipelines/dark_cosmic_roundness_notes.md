# Dark cosmic roundness workflow notes

## Why you saw the formatter error

Your stack failed to read Butler `Plot` storage-class outputs via
`MatplotlibFormatter`. To avoid this compatibility issue, the task now:

1. saves PNG cutouts directly to disk, and
2. writes a Butler `ArrowAstropy` manifest table listing those PNG paths.

## Browseable cutout folder

`artifacts/round_fat_dark_cutouts/<instrument>/exposure_<exposure>/detector_<detector>/`

## Butler queries for generated products

```python
from lsst.daf.butler import Butler

butler = Butler("/repo/main", collections=["u/gmegias/transient_catalogs", "LSSTCam/defaults"])

# Per-detector source catalog from detection
cat = butler.get(
    "detector_dark_source_catalog",
    instrument="LSSTCam",
    exposure=2025061300222,
    detector=42,
)

# Per-detector round+fat subset used for exports
round_fat_cat = butler.get(
    "detector_round_fat_dark_source_catalog",
    instrument="LSSTCam",
    exposure=2025061300222,
    detector=42,
)

# Per-detector manifest (contains PNG paths)
manifest = butler.get(
    "detector_round_fat_dark_source_manifest",
    instrument="LSSTCam",
    exposure=2025061300222,
    detector=42,
)

print(manifest["png_path"])
```

Then open those PNG files directly from the filesystem/Jupyter.
