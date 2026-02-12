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
from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure
from lsst.pipe.base import connectionTypes


class ExportRoundestDarkSourcesConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure", "detector"),
):
    inputCatalog = connectionTypes.Input(
        doc="Input per-detector dark source catalog.",
        name="detector_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )
    inputExp = connectionTypes.Input(
        doc="Input post-ISR dark exposure.",
        name="post_isr_dark",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    roundestCatalog = connectionTypes.Output(
        doc="Per-detector subset of round and fat sources used for cutout export.",
        name="detector_round_fat_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )
    roundestManifest = connectionTypes.Output(
        doc="Per-detector manifest with PNG paths for exported cutouts.",
        name="detector_round_fat_dark_source_manifest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )


class ExportRoundestDarkSourcesConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ExportRoundestDarkSourcesConnections,
):
    nSources = pexConfig.Field(dtype=int, default=15, doc="Maximum number of cutouts to export.")
    minRoundness = pexConfig.Field(dtype=float, default=0.0, doc="Minimum roundness threshold.")
    minDiameterPixels = pexConfig.Field(dtype=float, default=0.0, doc="Minimum equivalent diameter threshold.")
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
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(
            inputCatalog=inputs["inputCatalog"],
            inputExp=inputs["inputExp"],
            instrument=inputRefs.inputCatalog.dataId["instrument"],
            exposure=inputRefs.inputCatalog.dataId["exposure"],
            detector=inputRefs.inputCatalog.dataId["detector"],
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

    def _save_cutouts_and_build_manifest(
        self, selected, inputExp: Exposure, instrument: str, exposure: int, detector: int
    ) -> Table:
        out_dir = (
            Path(self.config.outputDirectory)
            / str(instrument)
            / f"exposure_{exposure}"
            / f"detector_{detector}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        image_array = inputExp.image.array

        source_ids = []
        roundness_values = []
        diameters = []
        relpaths = []

        for row in selected:
            cutout = self._extract_cutout(
                image_array=image_array,
                x=float(row["centroid_x"]),
                y=float(row["centroid_y"]),
                diameter=float(row["equivalent_diameter"]),
            )
            if cutout.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            vmin, vmax = np.nanpercentile(cutout, [5, 99])
            ax.imshow(cutout, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"r={row['roundness']:.2f}, d={row['equivalent_diameter']:.1f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()

            filename = f"src{int(row['id'])}_r{row['roundness']:.3f}.png"
            out_path = out_dir / filename
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            source_ids.append(int(row["id"]))
            roundness_values.append(float(row["roundness"]))
            diameters.append(float(row["equivalent_diameter"]))
            relpaths.append(str(out_path))

        return Table(
            {
                "source_id": source_ids,
                "roundness": roundness_values,
                "equivalent_diameter": diameters,
                "png_path": relpaths,
            }
        )

    def run(self, inputCatalog, inputExp: Exposure, instrument: str, exposure: int, detector: int):  # type: ignore[no-untyped-def]
        selected = self._select(inputCatalog)
        manifest = self._save_cutouts_and_build_manifest(selected, inputExp, instrument, exposure, detector)
        return pipeBase.Struct(roundestCatalog=selected, roundestManifest=manifest)
