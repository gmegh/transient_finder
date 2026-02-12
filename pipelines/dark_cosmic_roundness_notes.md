# Wavelet dark-ring workflow notes (per-visit)

## How roundness is estimated now

We now track two shape metrics per footprint:

1. **circularity** = `4πA / P²` computed on the hole-filled mask boundary, then clipped to `[0, 1]`.
   - This avoids the unphysical `>1` values you observed from pixelized perimeter approximations.
2. **gaussian_like_score** = normalized correlation of the (negative) cutout signal with an
   isotropic Gaussian model matched to the footprint centroid/width.

For final "interesting" candidates we require both:

- `circularity >= minCircularity`
- `gaussian_like_score >= minGaussianLike`

and then rank by `circularity * gaussian_like_score`.

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
- `stats/area_vs_circularity.png`
- `stats/circularity_hist.png`
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

print(visit_cat["circularity", "gaussian_like_score", "area"][:10])
print(roundest["detector", "footprint_id", "circularity", "gaussian_like_score", "selected_cutout_path"][:20])
print(stats_manifest["kind", "path"])
```
