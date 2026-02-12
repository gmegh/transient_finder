# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "ConsolidateDarkRingVisitStatsConnections",
    "ConsolidateDarkRingVisitStatsConfig",
    "ConsolidateDarkRingVisitStatsTask",
]

import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.coaddBase import reorderRefs


class ConsolidateDarkRingVisitStatsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    inputCatalogs = connectionTypes.Input(
        doc="Per-detector dark ring catalogs for one exposure.",
        name="detector_dark_ring_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    inputExps = connectionTypes.Input(
        doc="Per-detector post-ISR dark exposures for this visit/exposure.",
        name="post_isr_dark",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )

    outputVisitCatalog = connectionTypes.Output(
        doc="Exposure-level concatenated dark ring catalog.",
        name="exposure_dark_ring_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )
    outputVisitRoundest = connectionTypes.Output(
        doc="Exposure-level Gaussian-like candidates, ordered by score.",
        name="exposure_dark_ring_roundest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )
    outputVisitManifest = connectionTypes.Output(
        doc="Exposure-level manifest (plots + ordered roundest cutout paths).",
        name="exposure_dark_ring_stats_manifest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )


class ConsolidateDarkRingVisitStatsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ConsolidateDarkRingVisitStatsConnections,
):
    minGaussianLike = pexConfig.Field(dtype=float, default=0.7, doc="Minimum Gaussian-like score for selected candidates.")
    minGaussianDiameterPixels = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Minimum fitted Gaussian FWHM diameter (pixels) for selected candidates.",
    )
    referenceDiameterPixels = pexConfig.Field(
        dtype=float,
        default=9.0,
        doc="Reference PSF-like diameter (pixels) drawn in red on stats plots.",
    )
    mosaicMax = pexConfig.Field(dtype=int, default=100, doc="Maximum number of ordered cutouts shown in the visit super-plot.")
    outputRoot = pexConfig.Field(
        dtype=str,
        default="artifacts/dark_ring_wavelet_visit",
        doc="Root directory for per-exposure summary plots.",
    )


class ConsolidateDarkRingVisitStatsTask(pipeBase.PipelineTask):
    ConfigClass = ConsolidateDarkRingVisitStatsConfig
    _DefaultName = "consolidateDarkRingVisitStats"

    def _concat(self, handles):  # type: ignore[no-untyped-def]
        tables = [h.get() for h in handles]
        return vstack(tables, metadata_conflicts="silent") if tables else Table()

    def _extract(self, image: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        y0 = max(0, y0)
        y1 = min(image.shape[0], y1)
        x0 = max(0, x0)
        x1 = min(image.shape[1], x1)
        return image[y0:y1, x0:x1]

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.inputExps]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")

        inputs = butlerQC.get(inputRefs)
        catalog = self._concat(inputs["inputCatalogs"])

        detector_to_exp = {ref.dataId["detector"]: exp for ref, exp in zip(inputRefs.inputExps, inputs["inputExps"])}

        instrument = inputRefs.inputExps[0].dataId["instrument"] if inputRefs.inputExps else "unknown"
        exposure = inputRefs.inputExps[0].dataId["exposure"] if inputRefs.inputExps else -1

        out_dir = Path(self.config.outputRoot) / str(instrument) / f"exposure_{exposure}"
        stats_dir = out_dir / "stats"
        round_dir = out_dir / "roundest"
        stats_dir.mkdir(parents=True, exist_ok=True)
        round_dir.mkdir(parents=True, exist_ok=True)

        stats_rows: list[dict[str, str | int]] = []

        if len(catalog) > 0:
            gaus = np.asarray(catalog["gaussian_like_score"])
            area = np.asarray(catalog["area"])
            circ = np.asarray(catalog["circularity"])
            gaus_diam = np.asarray(catalog["gaussian_diameter_pixels"])
            finite = np.isfinite(gaus) & np.isfinite(area)

            fig, ax = plt.subplots(figsize=(7, 4))
            sc = ax.scatter(area[finite], gaus[finite], c=circ[finite], s=6, alpha=0.35, cmap="viridis")
            ax.set_xlabel("footprint area [pix]")
            ax.set_ylabel("gaussian_like_score")
            ax.set_title("Per-visit candidates: area vs gaussian-like")
            fig.colorbar(sc, ax=ax, label="circularity (diagnostic)")
            ref_area = np.pi * (0.5 * self.config.referenceDiameterPixels) ** 2
            ax.scatter([ref_area], [1.0], color="red", marker="x", s=80, label="9px perfect-PSF ref")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            p_sc = stats_dir / "area_vs_gaussian_like.png"
            fig.savefig(p_sc, dpi=140, bbox_inches="tight")
            plt.close(fig)
            stats_rows.append({"kind": "scatter", "path": str(p_sc), "footprint_id": -1})

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(gaus[np.isfinite(gaus)], bins=80, alpha=0.85, color="tab:blue")
            ax.set_xlabel("gaussian_like_score")
            ax.set_ylabel("N footprints")
            ax.set_title("Per-visit gaussian-like histogram")
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="perfect Gaussian ref")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            p_hist = stats_dir / "gaussian_like_hist.png"
            fig.savefig(p_hist, dpi=140, bbox_inches="tight")
            plt.close(fig)
            stats_rows.append({"kind": "histogram", "path": str(p_hist), "footprint_id": -1})

            select = (
                np.isfinite(gaus)
                & np.isfinite(gaus_diam)
                & (gaus >= self.config.minGaussianLike)
                & (gaus_diam >= self.config.minGaussianDiameterPixels)
            )
            roundest = catalog[select]
            if len(roundest) > 0:
                score = np.asarray(roundest["gaussian_like_score"])
                order = np.argsort(score)[::-1]
                roundest = roundest[order]

                selected_paths = []
                mosaic_pairs = []
                for rank, row in enumerate(roundest):
                    det = int(row["detector"])
                    exp = detector_to_exp.get(det)
                    if exp is None:
                        selected_paths.append("")
                        continue
                    stamp = self._extract(
                        exp.image.array,
                        int(row["x0"]),
                        int(row["x1"]),
                        int(row["y0"]),
                        int(row["y1"]),
                    )
                    if stamp.size == 0:
                        selected_paths.append("")
                        continue

                    out_png = round_dir / f"rank_{rank:05d}_det{det}_fp{int(row['footprint_id']):05d}_g{float(row['gaussian_like_score']):.3f}.png"
                    fig, ax = plt.subplots(figsize=(2.8, 2.8))
                    vmax = np.nanmax(np.abs(stamp))
                    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                    ax.imshow(stamp, vmin=-vmax, vmax=vmax, cmap="gray_r", origin="lower")
                    ax.set_title(f"rank={rank} g={float(row['gaussian_like_score']):.3f}", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.tight_layout()
                    fig.savefig(out_png, dpi=140, bbox_inches="tight")
                    plt.close(fig)

                    selected_paths.append(str(out_png))
                    stats_rows.append({"kind": "roundest_cutout", "path": str(out_png), "footprint_id": int(row["footprint_id"])})
                    if len(mosaic_pairs) < self.config.mosaicMax:
                        mosaic_pairs.append((rank, float(row["gaussian_like_score"]), stamp))

                roundest["selected_cutout_path"] = selected_paths

                if len(mosaic_pairs) > 0:
                    n = len(mosaic_pairs)
                    n_cols = min(10, n)
                    n_rows = int(math.ceil(n / n_cols))
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)
                    axes_flat = axes.flatten()
                    for i, (rank, score, stamp) in enumerate(mosaic_pairs):
                        ax = axes_flat[i]
                        vmax = np.nanmax(np.abs(stamp))
                        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                        ax.imshow(stamp, vmin=-vmax, vmax=vmax, cmap="gray_r", origin="lower")
                        ax.set_title(f"#{rank} g={score:.3f}", fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    for j in range(len(mosaic_pairs), len(axes_flat)):
                        axes_flat[j].set_axis_off()
                    fig.suptitle("Per-visit ordered roundest (gaussian-like) cutouts", fontsize=11)
                    fig.tight_layout()
                    p_mosaic = stats_dir / "roundest_mosaic.png"
                    fig.savefig(p_mosaic, dpi=140, bbox_inches="tight")
                    plt.close(fig)
                    stats_rows.append({"kind": "roundest_mosaic", "path": str(p_mosaic), "footprint_id": -1})
            else:
                roundest = Table(roundest)
        else:
            roundest = Table()

        out_manifest = Table(rows=stats_rows) if stats_rows else Table(
            names=["kind", "path", "footprint_id"], dtype=[str, str, int]
        )

        butlerQC.put(
            pipeBase.Struct(
                outputVisitCatalog=catalog,
                outputVisitRoundest=roundest,
                outputVisitManifest=out_manifest,
            ),
            outputRefs,
        )
