# Wavelet dark-ring workflow notes

This workflow no longer uses `SourceDetectionTask`. Instead it does:

1. starlet transform on `post_isr_dark`
2. threshold on a selected coefficient plane (default plane 2, threshold `< -3`)
3. connected-footprint extraction
4. circularity scoring + cutout/stat plot export

## Output folders

For each `(instrument, exposure, detector)`:

`artifacts/dark_ring_wavelet/<instrument>/exposure_<exposure>/detector_<detector>/`

Inside it:
- `all_cutouts/` : PNGs for all detected footprints (capped by `maxAllCutouts`)
- `stats/circularity_hist.png` : circularity histogram
- `roundest_cutouts/` : top `nRoundest` circularity cutouts

## Butler datasets

- `detector_dark_ring_catalog` (`ArrowAstropy`) with per-footprint metrics.
- `detector_dark_ring_manifest` (`ArrowAstropy`) with rows mapping product type to PNG file path.

```python
from lsst.daf.butler import Butler

butler = Butler("/repo/main", collections=["u/gmegias/transient_catalogs", "LSSTCam/defaults"])

cat = butler.get(
    "detector_dark_ring_catalog",
    instrument="LSSTCam",
    exposure=2025061300222,
    detector=42,
)

manifest = butler.get(
    "detector_dark_ring_manifest",
    instrument="LSSTCam",
    exposure=2025061300222,
    detector=42,
)

print(cat[:5])
print(manifest["kind", "path"])
```

Then open the PNG paths from `manifest["path"]` directly in Jupyter/file browser.
