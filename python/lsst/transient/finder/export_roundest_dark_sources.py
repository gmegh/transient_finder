# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "ExportRoundestDarkSourcesConnections",
    "ExportRoundestDarkSourcesConfig",
    "ExportRoundestDarkSourcesTask",
]

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.coaddBase import reorderRefs


class ExportRoundestDarkSourcesConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    inputCatalog = connectionTypes.Input(
        doc="Input per-exposure dark source catalog.",
        name="exposure_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )
    inputExposures = connectionTypes.Input(
        doc="Input post-ISR dark exposures for all detectors in this exposure.",
        name="post_isr_dark",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    roundestCatalog = connectionTypes.Output(
        doc="Per-exposure subset of round and fat sources used for cutout export.",
        name="exposure_round_fat_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )
    roundestPlot = connectionTypes.Output(
        doc="Per-exposure montage of selected round and fat dark-source cutouts.",
        name="exposure_round_fat_dark_source_plot",
        storageClass="Plot",
        dimensions=("instrument", "exposure"),
    )


class ExportRoundestDarkSourcesConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ExportRoundestDarkSourcesConnections,
):
    nSources = pexConfig.Field(dtype=int, default=25, doc="Maximum number of cutouts to export.")
    minRoundness = pexConfig.Field(dtype=float, default=0.85, doc="Minimum roundness threshold.")
    minDiameterPixels = pexConfig.Field(dtype=float, default=6.0, doc="Minimum equivalent diameter threshold.")
    padPixels = pexConfig.Field(dtype=int, default=2, doc="Padding around each source cutout.")
    outputDirectory = pexConfig.Field(
        dtype=str,
        default="artifacts/round_fat_dark_cutouts",
        doc="Directory where per-source PNG cutouts are saved.",
    )


class ExportRoundestDarkSourcesTask(pipeBase.PipelineTask):
    ConfigClass = ExportRoundestDarkSourcesConfig
    _DefaultName = "exportRoundestDarkSources"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.inputExposures]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")
        inputs = butlerQC.get(inputRefs)
        detector_to_exposure = {
            ref.dataId["detector"]: exp for ref, exp in zip(inputRefs.inputExposures, inputs["inputExposures"])
        }
        outputs = self.run(
            inputCatalog=inputs["inputCatalog"],
            detectorToExposure=detector_to_exposure,
            instrument=inputRefs.inputCatalog.dataId["instrument"],
            exposure=inputRefs.inputCatalog.dataId["exposure"],
        )
        butlerQC.put(outputs, outputRefs)

    def _select(self, inputCatalog):  # type: ignore[no-untyped-def]
        if len(inputCatalog) == 0:
            return inputCatalog
        mask = (
            np.isfinite(np.asarray(inputCatalog["roundness"]))
            & np.isfinite(np.asarray(inputCatalog["equivalent_diameter"]))
            & (np.asarray(inputCatalog["roundness"]) >= self.config.minRoundness)
            & (np.asarray(inputCatalog["equivalent_diameter"]) >= self.config.minDiameterPixels)
        )
        selected = inputCatalog[mask]
        if len(selected) == 0:
            return selected
        order = np.lexsort(
            (
                -np.asarray(selected["equivalent_diameter"]),
                -np.asarray(selected["roundness"]),
            )
        )
        return selected[order][: min(self.config.nSources, len(selected))]

    def _extract_cutout(self, image_array: np.ndarray, x: float, y: float, diameter: float) -> np.ndarray:
        radius = max(3, int(math.ceil(0.5 * diameter)) + self.config.padPixels)
        x_c = int(round(x))
        y_c = int(round(y))
        y0 = max(0, y_c - radius)
        y1 = min(image_array.shape[0], y_c + radius + 1)
        x0 = max(0, x_c - radius)
        x1 = min(image_array.shape[1], x_c + radius + 1)
        return image_array[y0:y1, x0:x1]

    def _save_cutouts(
        self,
        selected,
        detectorToExposure: dict[int, Exposure],
        instrument: str,
        exposure: int,
    ) -> None:
        out_dir = Path(self.config.outputDirectory) / str(instrument) / f"exposure_{exposure}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            detector = int(row["detector"])
            exp = detectorToExposure.get(detector)
            if exp is None:
                continue
            cutout = self._extract_cutout(
                image_array=exp.image.array,
                x=float(row["centroid_x"]),
                y=float(row["centroid_y"]),
                diameter=float(row["equivalent_diameter"]),
            )
            if cutout.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            vmin, vmax = np.nanpercentile(cutout, [5, 99])
            ax.imshow(cutout, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"det={detector} r={row['roundness']:.2f} d={row['equivalent_diameter']:.1f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            out_path = out_dir / f"det{detector}_src{int(row['id'])}_r{row['roundness']:.3f}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    def run(self, inputCatalog, detectorToExposure: dict[int, Exposure], instrument: str, exposure: int):  # type: ignore[no-untyped-def]
        selected = self._select(inputCatalog)
        self._save_cutouts(selected, detectorToExposure, instrument, exposure)

        if len(selected) == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(
                0.5,
                0.5,
                (
                    "No sources passing cuts\n"
                    f"roundness >= {self.config.minRoundness}, "
                    f"equivalent_diameter >= {self.config.minDiameterPixels:.1f}px"
                ),
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            return pipeBase.Struct(roundestCatalog=selected, roundestPlot=fig)

        rows = []
        for row in selected:
            detector = int(row["detector"])
            exp = detectorToExposure.get(detector)
            if exp is None:
                continue
            cutout = self._extract_cutout(
                image_array=exp.image.array,
                x=float(row["centroid_x"]),
                y=float(row["centroid_y"]),
                diameter=float(row["equivalent_diameter"]),
            )
            if cutout.size > 0:
                rows.append((row, cutout))

        if len(rows) == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, "No usable cutouts", ha="center", va="center")
            ax.set_axis_off()
            return pipeBase.Struct(roundestCatalog=selected, roundestPlot=fig)

        n_sources = len(rows)
        n_cols = min(5, n_sources)
        n_rows = int(math.ceil(n_sources / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for i, (row, cutout) in enumerate(rows):
            ax = axes_flat[i]
            vmin, vmax = np.nanpercentile(cutout, [5, 99])
            ax.imshow(cutout, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"det={int(row['detector'])}, r={row['roundness']:.2f}, d={row['equivalent_diameter']:.1f}px",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n_sources, len(axes_flat)):
            axes_flat[j].set_axis_off()

        fig.suptitle("Per-exposure round + fat dark-source cutouts", fontsize=12)
        fig.tight_layout()
        return pipeBase.Struct(roundestCatalog=selected, roundestPlot=fig)
