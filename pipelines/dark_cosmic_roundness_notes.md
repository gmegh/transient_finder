# Wavelet dark-ring workflow notes (per-visit)

## How roundness is estimated now

We now track two shape metrics per footprint:

1. **circularity** = `4πA / P²` computed on the hole-filled mask boundary, clipped to `[0, 1]`.
   - kept as a diagnostic only.
2. **gaussian_like_score** = normalized correlation of the (negative) cutout with a
   Gaussian model using an **adaptive sigma scan** around the moment-estimated sigma.
3. **gaussian_sigma_pixels** = best-fit sigma (pixels) that maximized the Gaussian correlation.

For final "interesting" candidates we now require only:

- `gaussian_like_score >= minGaussianLike`

and rank by `gaussian_like_score`.


## What to run first with BPS

1. **First pass (per-detector wavelet products):**
   run `pipelines/find_dark_sources_pipeline.yaml` (or the combined pipeline below).
2. **Then per-visit consolidation/stats:**
   run `pipelines/consolidate_dark_sources_pipeline.yaml`.

If you want one BPS submission that does both stages in order, use:
`pipelines/dark_cosmic_roundness_pipeline.yaml`.

## Workflow stages

1. `FindDarkRingsWaveletTask` (per detector):
   - starlet transform + threshold (`coeff[plane] < threshold`)
   - detect connected footprints
   - compute metrics and save **all cutouts**
2. `ConsolidateDarkRingVisitStatsTask` (per visit/exposure):
   - merge all detector catalogs for that visit
   - produce per-visit scatter/hist plots
   - select gaussian-like round candidates and export them in a visit folder

## Output folders

Per-detector all cutouts:

`artifacts/dark_ring_wavelet/<instrument>/exposure_<exposure>/detector_<detector>/all_cutouts/`

Per-visit stats + selected roundest:

`artifacts/dark_ring_wavelet_visit/<instrument>/exposure_<exposure>/`

with:
- `stats/area_vs_gaussian_like.png`
- `stats/gaussian_like_hist.png`
- `roundest/` (selected gaussian-like round cutouts)

## Butler datasets

Per-detector:
- `detector_dark_ring_catalog`
- `detector_dark_ring_manifest`

Per-visit:
- `exposure_dark_ring_catalog`
- `exposure_dark_ring_roundest`
- `exposure_dark_ring_stats_manifest`

```python
from lsst.daf.butler import Butler

butler = Butler("/repo/main", collections=["u/gmegias/transient_catalogs", "LSSTCam/defaults"])

visit_cat = butler.get(
    "exposure_dark_ring_catalog",
    instrument="LSSTCam",
    exposure=2025061300222,
)

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

print(visit_cat["gaussian_like_score", "gaussian_sigma_pixels", "circularity", "area"][:10])
print(roundest["detector", "footprint_id", "gaussian_like_score", "gaussian_sigma_pixels", "selected_cutout_path"][:20])
print(stats_manifest["kind", "path"])
```
