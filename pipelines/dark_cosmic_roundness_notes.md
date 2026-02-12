# Wavelet dark-ring workflow notes (per-visit)

## Key change

We no longer save all cutouts per detector by default.
Instead, we generate plots/cutouts at the **per-visit** stage, ordered by gaussian-likeness.

## How Gaussian matching works now (adaptive size)

For each detected footprint stamp we:

1. Build a negative-signal image (`-stamp`, clipped above background).
2. Estimate a base sigma from second moments.
3. Scan multiple sigma scales around that estimate (`0.6x, 0.8x, 1.0x, 1.25x, 1.6x`).
4. Keep the **best** normalized correlation as `gaussian_like_score` and its sigma as
   `gaussian_sigma_pixels`.

Circularity is still recorded as a diagnostic, but selection/ranking is gaussian-only.

## What to run first with BPS

1. **Per-detector wavelet detection catalogs**
   - run: `pipelines/find_dark_sources_pipeline.yaml`
2. **Per-visit stats and ordered roundest plots/cutouts**
   - run: `pipelines/consolidate_dark_sources_pipeline.yaml`

If you want one run doing both stages in order, use:
- `pipelines/dark_cosmic_roundness_pipeline.yaml`

## Outputs

Per-visit folder:

`artifacts/dark_ring_wavelet_visit/<instrument>/exposure_<exposure>/`

Includes:
- `stats/area_vs_gaussian_like.png`
- `stats/gaussian_like_hist.png`
- `stats/roundest_mosaic.png`  ← super plot ordered by roundness/gaussian score
- `roundest/` ordered cutout PNGs (`rank_00000_...`, `rank_00001_...`, ...)

## Butler datasets

Per-detector stage:
- `detector_dark_ring_catalog`

Per-visit stage:
- `exposure_dark_ring_catalog`
- `exposure_dark_ring_roundest`
- `exposure_dark_ring_stats_manifest`

```python
from lsst.daf.butler import Butler

butler = Butler("/repo/main", collections=["u/gmegias/transient_catalogs", "LSSTCam/defaults"])

roundest = butler.get(
    "exposure_dark_ring_roundest",
    instrument="LSSTCam",
    exposure=2025061300222,
)

stats_manifest = butler.get(
    "exposure_dark_ring_stats_manifest",
    instrument="LSSTCam",
    exposure=2025061300222,
)

print(roundest["detector", "footprint_id", "gaussian_like_score", "gaussian_sigma_pixels", "selected_cutout_path"][:20])
print(stats_manifest["kind", "path"])
```
