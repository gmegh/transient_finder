# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "FindDarkRingsWaveletConnections",
    "FindDarkRingsWaveletConfig",
    "FindDarkRingsWaveletTask",
]

import math
from pathlib import Path

import matplotlib.pyplot as plt
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

    ringManifest = cT.Output(
        name="detector_dark_ring_manifest",
        doc="Per-detector PNG manifest for all/roundest cutouts and histograms.",
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
    minFootprintArea = pexConfig.Field(dtype=int, default=30, doc="Minimum footprint area in pixels.")
    maxFootprintArea = pexConfig.Field(dtype=int, default=1000, doc="Maximum footprint area in pixels.")
    circularityThreshold = pexConfig.Field(dtype=float, default=0.9, doc="Minimum circularity for ring-like flag.")
    minHoleFraction = pexConfig.Field(dtype=float, default=0.05, doc="Minimum hole area / bbox area.")
    maxHoleFraction = pexConfig.Field(dtype=float, default=0.5, doc="Maximum hole area / filled area.")
    centeringThreshold = pexConfig.Field(dtype=float, default=0.4, doc="Maximum normalized offset of hole centroid.")
    maxAllCutouts = pexConfig.Field(dtype=int, default=300, doc="Maximum number of all-candidate cutouts to save.")
    nRoundest = pexConfig.Field(dtype=int, default=40, doc="Number of roundest cutouts to save.")
    outputRoot = pexConfig.Field(dtype=str, default="artifacts/dark_ring_wavelet", doc="Root directory for PNG outputs.")


class FindDarkRingsWaveletTask(pipeBase.PipelineTask):
    ConfigClass = FindDarkRingsWaveletConfig
    _DefaultName = "findDarkRingsWavelet"

    def _compute_circularity(self, mask: np.ndarray) -> tuple[bool, float, float, float]:
        if int(mask.sum()) < 8:
            return False, np.nan, np.nan, np.nan

        filled = ndimage.binary_fill_holes(mask)
        holes = filled & ~mask
        if int(holes.sum()) == 0:
            return False, 0.0, 0.0, 0.0

        bbox_area = float(mask.shape[0] * mask.shape[1])
        total_area = float(filled.sum())
        hole_area = float(holes.sum())
        hole_frac_bbox = hole_area / bbox_area if bbox_area > 0 else np.nan
        hole_frac_total = hole_area / total_area if total_area > 0 else np.nan

        if hole_frac_bbox < self.config.minHoleFraction or hole_frac_total > self.config.maxHoleFraction:
            return False, 0.0, hole_frac_bbox, hole_frac_total

        mask_centroid = np.array(ndimage.center_of_mass(filled))
        hole_centroid = np.array(ndimage.center_of_mass(holes))
        diag = math.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
        offset = float(np.linalg.norm(hole_centroid - mask_centroid) / diag) if diag > 0 else np.inf
        if offset > self.config.centeringThreshold:
            return False, 0.0, hole_frac_bbox, hole_frac_total

        eroded = ndimage.binary_erosion(filled)
        perimeter = float(np.sum(filled & ~eroded))
        if perimeter <= 0:
            return False, 0.0, hole_frac_bbox, hole_frac_total

        circularity = float(4.0 * np.pi * total_area / perimeter**2)
        is_ring = circularity >= self.config.circularityThreshold
        return is_ring, circularity, hole_frac_bbox, hole_frac_total

    def _save_stamp(self, stamp: np.ndarray, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(3, 3))
        vmax = np.nanmax(np.abs(stamp)) if stamp.size > 0 else 1.0
        vmax = vmax if vmax > 0 else 1.0
        ax.imshow(stamp, vmin=-vmax, vmax=vmax, cmap="gray_r", origin="lower")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(
            inputExp=inputs["inputExp"],
            instrument=inputRefs.inputExp.dataId["instrument"],
            exposure=inputRefs.inputExp.dataId["exposure"],
            detector=inputRefs.inputExp.dataId["detector"],
        )
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp: Exposure, instrument: str, exposure: int, detector: int) -> pipeBase.Struct:
        try:
            import lsst.scarlet.lite as scl
        except ImportError as e:
            raise RuntimeError("lsst.scarlet.lite is required for wavelet dark-ring detection.") from e

        image = inputExp.image.array
        coeffs = scl.wavelet.starlet_transform(image, scales=self.config.scales)
        mask = coeffs[self.config.coefficientPlane] < self.config.negativeThreshold
        footprints = scl.detect_pybind11.get_footprints(
            mask,
            1,
            self.config.minFootprintArea,
            0,
            0,
            False,
            0,
            0,
        )

        base_dir = Path(self.config.outputRoot) / str(instrument) / f"exposure_{exposure}" / f"detector_{detector}"
        all_dir = base_dir / "all_cutouts"
        round_dir = base_dir / "roundest_cutouts"
        stats_dir = base_dir / "stats"
        for d in (all_dir, round_dir, stats_dir):
            d.mkdir(parents=True, exist_ok=True)

        rows = []
        manifest_rows = []

        for i, fp in enumerate(footprints):
            area = int(np.sum(fp.data))
            if area > self.config.maxFootprintArea:
                continue

            is_ring, circularity, hole_frac_bbox, hole_frac_total = self._compute_circularity(fp.data)
            y0, y1, x0, x1 = fp.bounds
            stamp = image[fp.bbox.slices]

            rows.append(
                {
                    "footprint_id": i,
                    "detector": detector,
                    "exposure": exposure,
                    "area": area,
                    "circularity": circularity,
                    "is_ring": bool(is_ring),
                    "hole_frac_bbox": hole_frac_bbox,
                    "hole_frac_total": hole_frac_total,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                }
            )

            if i < self.config.maxAllCutouts and stamp.size > 0:
                all_path = all_dir / f"fp_{i:05d}.png"
                self._save_stamp(stamp, all_path, f"fp={i} circ={circularity:.3f} area={area}")
                manifest_rows.append({"kind": "all_cutout", "footprint_id": i, "path": str(all_path)})

        catalog = Table(rows=rows) if rows else Table(
            names=[
                "footprint_id", "detector", "exposure", "area", "circularity", "is_ring", "hole_frac_bbox",
                "hole_frac_total", "y0", "y1", "x0", "x1"
            ],
            dtype=[int, int, int, int, float, bool, float, float, int, int, int, int],
        )

        if len(catalog) > 0:
            finite = np.isfinite(np.asarray(catalog["circularity"]))
            vals = np.asarray(catalog["circularity"])[finite]
            fig, ax = plt.subplots(figsize=(5, 3))
            if len(vals) > 0:
                ax.hist(vals, bins=40, color="tab:blue", alpha=0.9)
            ax.set_xlabel("circularity")
            ax.set_ylabel("N footprints")
            ax.set_title("Wavelet footprint circularity")
            fig.tight_layout()
            hist_path = stats_dir / "circularity_hist.png"
            fig.savefig(hist_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            manifest_rows.append({"kind": "histogram", "footprint_id": -1, "path": str(hist_path)})

            order = np.argsort(np.asarray(catalog["circularity"]))[::-1]
            round_ids = np.asarray(catalog["footprint_id"])[order[: self.config.nRoundest]]
            fp_by_id = {int(r["footprint_id"]): int(idx) for idx, r in enumerate(catalog)}
            for rank, fp_id in enumerate(round_ids):
                row = catalog[fp_by_id[int(fp_id)]]
                y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
                stamp = image[y0:y1, x0:x1]
                if stamp.size == 0:
                    continue
                p = round_dir / f"rank_{rank:03d}_fp_{int(fp_id):05d}.png"
                self._save_stamp(stamp, p, f"rank={rank} fp={int(fp_id)} circ={float(row['circularity']):.3f}")
                manifest_rows.append({"kind": "roundest_cutout", "footprint_id": int(fp_id), "path": str(p)})

        manifest = Table(rows=manifest_rows) if manifest_rows else Table(
            names=["kind", "footprint_id", "path"],
            dtype=[str, int, str],
        )

        return pipeBase.Struct(ringCatalog=catalog, ringManifest=manifest)
