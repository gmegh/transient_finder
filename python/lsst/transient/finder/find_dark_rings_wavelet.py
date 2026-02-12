# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "FindDarkRingsWaveletConnections",
    "FindDarkRingsWaveletConfig",
    "FindDarkRingsWaveletTask",
]

import math

import numpy as np
from astropy.table import Table
from scipy import ndimage

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import Exposure


class FindDarkRingsWaveletConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure", "detector"),
):
    inputExp = cT.Input(
        name="post_isr_dark",
        doc="Input post-ISR dark exposure.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )

    ringCatalog = cT.Output(
        name="detector_dark_ring_catalog",
        doc="Per-detector ring candidate metrics from wavelet footprints.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )


class FindDarkRingsWaveletConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FindDarkRingsWaveletConnections,
):
    scales = pexConfig.Field(dtype=int, default=3, doc="Number of starlet scales.")
    coefficientPlane = pexConfig.Field(dtype=int, default=2, doc="Wavelet coefficient plane index to threshold.")
    negativeThreshold = pexConfig.Field(dtype=float, default=-3.0, doc="Negative threshold on selected wavelet plane.")
    minFootprintArea = pexConfig.Field(dtype=int, default=10, doc="Minimum footprint area in pixels.")
    maxFootprintArea = pexConfig.Field(dtype=int, default=1000, doc="Maximum footprint area in pixels.")
    minHoleFraction = pexConfig.Field(dtype=float, default=0.05, doc="Minimum hole area / bbox area.")
    maxHoleFraction = pexConfig.Field(dtype=float, default=0.5, doc="Maximum hole area / filled area.")
    centeringThreshold = pexConfig.Field(dtype=float, default=0.4, doc="Maximum normalized offset of hole centroid.")

    minGaussianDiameterPixels = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Minimum Gaussian-equivalent FWHM diameter (pixels) to keep candidate.",
    )


class FindDarkRingsWaveletTask(pipeBase.PipelineTask):
    ConfigClass = FindDarkRingsWaveletConfig
    _DefaultName = "findDarkRingsWavelet"

    def _compute_circularity(self, mask: np.ndarray) -> tuple[float, float, float]:
        filled = ndimage.binary_fill_holes(mask)
        holes = filled & ~mask

        bbox_area = float(mask.shape[0] * mask.shape[1])
        total_area = float(filled.sum())
        hole_area = float(holes.sum())
        hole_frac_bbox = hole_area / bbox_area if bbox_area > 0 else np.nan
        hole_frac_total = hole_area / total_area if total_area > 0 else np.nan

        if total_area <= 0:
            return 0.0, hole_frac_bbox, hole_frac_total

        boundary = filled ^ ndimage.binary_erosion(filled)
        perimeter = float(boundary.sum())
        if perimeter <= 0:
            return 0.0, hole_frac_bbox, hole_frac_total

        circularity = float(4.0 * np.pi * total_area / perimeter**2)
        return float(np.clip(circularity, 0.0, 1.0)), hole_frac_bbox, hole_frac_total

    def _gaussian_like_score(self, stamp: np.ndarray) -> tuple[float, float]:
        if stamp.size == 0:
            return 0.0, np.nan
        signal = -stamp.astype(float)
        signal -= np.nanmedian(signal)
        signal = np.clip(signal, 0.0, None)
        total = float(signal.sum())
        if total <= 0:
            return 0.0, np.nan

        y, x = np.indices(signal.shape)
        x0 = float((signal * x).sum() / total)
        y0 = float((signal * y).sum() / total)
        var_x = float((signal * (x - x0) ** 2).sum() / total)
        var_y = float((signal * (y - y0) ** 2).sum() / total)
        sigma0 = math.sqrt(max(0.5, 0.5 * (var_x + var_y)))

        s = signal.ravel() - signal.ravel().mean()
        s_norm = float(np.linalg.norm(s))
        if s_norm <= 0:
            return 0.0, np.nan

        best_corr = -1.0
        best_sigma = sigma0
        for factor in (0.6, 0.8, 1.0, 1.25, 1.6):
            sigma = max(0.5, sigma0 * factor)
            model = np.exp(-0.5 * (((x - x0) ** 2 + (y - y0) ** 2) / (sigma**2)))
            m = model.ravel() - model.ravel().mean()
            m_norm = float(np.linalg.norm(m))
            if m_norm <= 0:
                continue
            corr = float(np.dot(s, m) / (s_norm * m_norm))
            if corr > best_corr:
                best_corr = corr
                best_sigma = sigma

        return float(np.clip(best_corr, -1.0, 1.0)), float(best_sigma)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(
            inputExp=inputs["inputExp"],
            exposure=inputRefs.inputExp.dataId["exposure"],
            detector=inputRefs.inputExp.dataId["detector"],
        )
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp: Exposure, exposure: int, detector: int) -> pipeBase.Struct:
        try:
            import lsst.scarlet.lite as scl
        except ImportError as e:
            raise RuntimeError("lsst.scarlet.lite is required for wavelet dark-ring detection.") from e

        image = inputExp.image.array
        coeffs = scl.wavelet.starlet_transform(image, scales=self.config.scales)
        mask = coeffs[self.config.coefficientPlane] < self.config.negativeThreshold
        footprints = scl.detect_pybind11.get_footprints(mask, 1, self.config.minFootprintArea, 0, 0, False, 0, 0)

        rows = []
        for i, fp in enumerate(footprints):
            area = int(np.sum(fp.data))
            if area > self.config.maxFootprintArea or area < self.config.minFootprintArea:
                continue

            y0, y1, x0, x1 = fp.bounds
            stamp = image[fp.bbox.slices]
            if stamp.size == 0:
                continue

            circularity, hole_frac_bbox, hole_frac_total = self._compute_circularity(fp.data)
            gaussian_like, gaussian_sigma = self._gaussian_like_score(stamp)
            gaussian_diameter = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * gaussian_sigma) if np.isfinite(gaussian_sigma) else np.nan
            if (not np.isfinite(gaussian_diameter)) or gaussian_diameter < self.config.minGaussianDiameterPixels:
                continue

            filled = ndimage.binary_fill_holes(fp.data)
            holes = filled & ~fp.data
            mask_centroid = np.array(ndimage.center_of_mass(filled))
            hole_centroid = np.array(ndimage.center_of_mass(holes)) if holes.sum() > 0 else mask_centroid
            diag = math.sqrt(fp.data.shape[0] ** 2 + fp.data.shape[1] ** 2)
            offset = float(np.linalg.norm(hole_centroid - mask_centroid) / diag) if diag > 0 else np.inf
            passes_hole = (
                hole_frac_bbox >= self.config.minHoleFraction
                and hole_frac_total <= self.config.maxHoleFraction
                and offset <= self.config.centeringThreshold
            )

            rows.append(
                {
                    "footprint_id": i,
                    "detector": detector,
                    "exposure": exposure,
                    "area": area,
                    "circularity": circularity,
                    "gaussian_like_score": gaussian_like,
                    "gaussian_sigma_pixels": gaussian_sigma,
                    "gaussian_diameter_pixels": gaussian_diameter,
                    "passes_hole_checks": bool(passes_hole),
                    "hole_frac_bbox": hole_frac_bbox,
                    "hole_frac_total": hole_frac_total,
                    "center_offset": offset,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                }
            )

        catalog = Table(rows=rows) if rows else Table(
            names=[
                "footprint_id", "detector", "exposure", "area", "circularity", "gaussian_like_score",
                "gaussian_sigma_pixels", "gaussian_diameter_pixels", "passes_hole_checks", "hole_frac_bbox", "hole_frac_total",
                "center_offset", "y0", "y1", "x0", "x1",
            ],
            dtype=[int, int, int, int, float, float, float, float, bool, float, float, float, int, int, int, int],
        )

        return pipeBase.Struct(ringCatalog=catalog)
