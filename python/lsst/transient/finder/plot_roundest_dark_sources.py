# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "PlotRoundestDarkSourcesConnections",
    "PlotRoundestDarkSourcesConfig",
    "PlotRoundestDarkSourcesTask",
]

import math

import matplotlib.pyplot as plt
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure
from lsst.pipe.base import connectionTypes


class PlotRoundestDarkSourcesConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure", "detector"),
):
    inputExp = connectionTypes.Input(
        doc="Input post-ISR dark exposure.",
        name="post_isr_dark",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    inputCatalog = connectionTypes.Input(
        doc="Input per-detector dark source catalog.",
        name="detector_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )
    roundestCatalog = connectionTypes.Output(
        doc="Round and fat dark sources for one detector exposure.",
        name="detector_round_fat_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )
    roundestPlot = connectionTypes.Output(
        doc="Cutout montage of round and fat dark-source footprints.",
        name="detector_round_fat_dark_source_plot",
        storageClass="Plot",
        dimensions=("instrument", "exposure", "detector"),
    )


class PlotRoundestDarkSourcesConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=PlotRoundestDarkSourcesConnections,
):
    nSources = pexConfig.Field(dtype=int, default=25, doc="Maximum number of sources to plot.")
    minRoundness = pexConfig.Field(dtype=float, default=0.8, doc="Minimum roundness for plotting.")
    minDiameterPixels = pexConfig.Field(
        dtype=float,
        default=6.0,
        doc="Minimum equivalent diameter in pixels for plotting.",
    )
    padPixels = pexConfig.Field(dtype=int, default=2, doc="Extra padding around source diameter in cutouts.")


class PlotRoundestDarkSourcesTask(pipeBase.PipelineTask):
    ConfigClass = PlotRoundestDarkSourcesConfig
    _DefaultName = "plotRoundestDarkSources"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(inputExp=inputs["inputExp"], inputCatalog=inputs["inputCatalog"])
        butlerQC.put(outputs, outputRefs)

    def _select_round_fat(self, inputCatalog):  # type: ignore[no-untyped-def]
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

        sort_idx = np.lexsort(
            (
                -np.asarray(selected["equivalent_diameter"]),
                -np.asarray(selected["roundness"]),
            )
        )
        sorted_selected = selected[sort_idx]
        return sorted_selected[: min(self.config.nSources, len(sorted_selected))]

    def _extract_cutout(self, image_array: np.ndarray, x: float, y: float, diameter: float) -> np.ndarray:
        radius = max(3, int(math.ceil(0.5 * diameter)) + self.config.padPixels)
        x_c = int(round(x))
        y_c = int(round(y))
        y0 = max(0, y_c - radius)
        y1 = min(image_array.shape[0], y_c + radius + 1)
        x0 = max(0, x_c - radius)
        x1 = min(image_array.shape[1], x_c + radius + 1)
        return image_array[y0:y1, x0:x1]

    def run(self, inputExp: Exposure, inputCatalog):  # type: ignore[no-untyped-def]
        selected = self._select_round_fat(inputCatalog)

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

        image_array = inputExp.image.array
        n_sources = len(selected)
        n_cols = min(5, n_sources)
        n_rows = int(math.ceil(n_sources / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for i, row in enumerate(selected):
            cutout = self._extract_cutout(
                image_array=image_array,
                x=float(row["centroid_x"]),
                y=float(row["centroid_y"]),
                diameter=float(row["equivalent_diameter"]),
            )
            ax = axes_flat[i]
            if cutout.size == 0:
                ax.text(0.5, 0.5, "edge", ha="center", va="center")
                ax.set_axis_off()
                continue
            vmin, vmax = np.nanpercentile(cutout, [5, 99])
            ax.imshow(cutout, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"r={row['roundness']:.2f}, d={row['equivalent_diameter']:.1f}px",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n_sources, len(axes_flat)):
            axes_flat[j].set_axis_off()

        fig.suptitle("Round + fat dark-source footprint cutouts", fontsize=12)
        fig.tight_layout()
        return pipeBase.Struct(roundestCatalog=selected, roundestPlot=fig)
