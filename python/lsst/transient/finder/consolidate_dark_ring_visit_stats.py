# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "ConsolidateDarkRingVisitStatsConnections",
    "ConsolidateDarkRingVisitStatsConfig",
    "ConsolidateDarkRingVisitStatsTask",
]

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes


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
    inputManifests = connectionTypes.Input(
        doc="Per-detector dark ring manifests for one exposure.",
        name="detector_dark_ring_manifest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )

    outputVisitCatalog = connectionTypes.Output(
        doc="Exposure-level concatenated dark ring catalog.",
        name="exposure_dark_ring_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )
    outputVisitRoundest = connectionTypes.Output(
        doc="Exposure-level Gaussian-like round candidates.",
        name="exposure_dark_ring_roundest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )
    outputVisitManifest = connectionTypes.Output(
        doc="Exposure-level manifest (plots + selected cutout paths).",
        name="exposure_dark_ring_stats_manifest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )


class ConsolidateDarkRingVisitStatsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ConsolidateDarkRingVisitStatsConnections,
):
    minGaussianLike = pexConfig.Field(dtype=float, default=0.7, doc="Minimum Gaussian-like score for selected candidates.")
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

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        inputs = butlerQC.get(inputRefs)
        catalog = self._concat(inputs["inputCatalogs"])
        manifest = self._concat(inputs["inputManifests"])

        instrument = inputRefs.inputCatalogs[0].dataId["instrument"] if inputRefs.inputCatalogs else "unknown"
        exposure = inputRefs.inputCatalogs[0].dataId["exposure"] if inputRefs.inputCatalogs else -1

        out_dir = Path(self.config.outputRoot) / str(instrument) / f"exposure_{exposure}"
        stats_dir = out_dir / "stats"
        round_dir = out_dir / "roundest"
        stats_dir.mkdir(parents=True, exist_ok=True)
        round_dir.mkdir(parents=True, exist_ok=True)

        stats_rows: list[dict[str, str | int]] = []

        if len(catalog) > 0:
            circ = np.asarray(catalog["circularity"])
            gaus = np.asarray(catalog["gaussian_like_score"])
            area = np.asarray(catalog["area"])
            finite = np.isfinite(circ) & np.isfinite(gaus) & np.isfinite(area)

            fig, ax = plt.subplots(figsize=(7, 4))
            sc = ax.scatter(area[finite], gaus[finite], c=circ[finite], s=5, alpha=0.35, cmap="viridis")
            ax.set_xlabel("footprint area [pix]")
            ax.set_ylabel("gaussian_like_score")
            ax.set_title("Per-visit candidates: area vs gaussian-like")
            fig.colorbar(sc, ax=ax, label="circularity")
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
            fig.tight_layout()
            p_hist = stats_dir / "gaussian_like_hist.png"
            fig.savefig(p_hist, dpi=140, bbox_inches="tight")
            plt.close(fig)
            stats_rows.append({"kind": "histogram", "path": str(p_hist), "footprint_id": -1})

            select = np.isfinite(gaus) & (gaus >= self.config.minGaussianLike)
            selected = catalog[select]
            if len(selected) > 0:
                score = np.asarray(selected["gaussian_like_score"])
                order = np.argsort(score)[::-1]
                roundest = selected[order]
            else:
                roundest = selected

            # attach all_cutout path for selected rows and mirror into roundest folder
            if len(roundest) > 0 and len(manifest) > 0:
                all_cuts = manifest[manifest["kind"] == "all_cutout"]
                key_to_path = {(int(r["footprint_id"]), int(r["detector"])): str(r["path"]) for r in all_cuts}
                paths = []
                for r in roundest:
                    p = key_to_path.get((int(r["footprint_id"]), int(r["detector"])), "")
                    paths.append(p)
                    if p:
                        src = Path(p)
                        if src.exists():
                            dst = round_dir / src.name
                            if not dst.exists():
                                shutil.copy2(src, dst)
                            stats_rows.append(
                                {
                                    "kind": "roundest_cutout",
                                    "path": str(dst),
                                    "footprint_id": int(r["footprint_id"]),
                                }
                            )
                roundest["selected_cutout_path"] = paths
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
