# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "ConsolidateDarkRingStatsConnections",
    "ConsolidateDarkRingStatsConfig",
    "ConsolidateDarkRingStatsTask",
]

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes


class ConsolidateDarkRingStatsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument",),
):
    inputCatalogs = connectionTypes.Input(
        doc="Per-detector dark ring catalogs across exposures.",
        name="detector_dark_ring_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    inputManifests = connectionTypes.Input(
        doc="Per-detector dark ring manifests across exposures.",
        name="detector_dark_ring_manifest",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )

    outputAllCatalog = connectionTypes.Output(
        doc="Instrument-level concatenated dark ring catalog.",
        name="instrument_dark_ring_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
    )
    outputTopCatalog = connectionTypes.Output(
        doc="Top roundest dark ring candidates across all processed exposures/detectors.",
        name="instrument_dark_ring_top_roundest",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
    )
    outputStatsManifest = connectionTypes.Output(
        doc="Manifest for instrument-level summary plots and artifact paths.",
        name="instrument_dark_ring_stats_manifest",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
    )


class ConsolidateDarkRingStatsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ConsolidateDarkRingStatsConnections,
):
    nRoundest = pexConfig.Field(dtype=int, default=200, doc="Number of top roundest candidates to retain.")
    outputRoot = pexConfig.Field(
        dtype=str,
        default="artifacts/dark_ring_wavelet_global",
        doc="Root directory for global summary plots.",
    )


class ConsolidateDarkRingStatsTask(pipeBase.PipelineTask):
    ConfigClass = ConsolidateDarkRingStatsConfig
    _DefaultName = "consolidateDarkRingStats"

    def _concat_handles(self, handles):  # type: ignore[no-untyped-def]
        tables = [h.get() for h in handles]
        return vstack(tables, metadata_conflicts="silent") if tables else Table()

    def _make_plots(self, catalog: Table, out_dir: Path) -> list[dict[str, str]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        if len(catalog) == 0:
            return rows

        circ = np.asarray(catalog["circularity"])
        area = np.asarray(catalog["area"])
        finite = np.isfinite(circ) & np.isfinite(area)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(area[finite], circ[finite], s=4, alpha=0.35)
        ax.set_xlabel("footprint area [pix]")
        ax.set_ylabel("circularity")
        ax.set_title("All dark-ring footprint candidates")
        ax.grid(alpha=0.2)
        scatter_path = out_dir / "all_candidates_area_vs_circularity.png"
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        rows.append({"kind": "scatter", "path": str(scatter_path)})

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(circ[np.isfinite(circ)], bins=80, color="tab:blue", alpha=0.9)
        ax.set_xlabel("circularity")
        ax.set_ylabel("N footprints")
        ax.set_title("Circularity distribution across all processed data")
        hist_path = out_dir / "all_candidates_circularity_hist.png"
        fig.tight_layout()
        fig.savefig(hist_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        rows.append({"kind": "histogram", "path": str(hist_path)})

        return rows

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        inputs = butlerQC.get(inputRefs)
        all_catalog = self._concat_handles(inputs["inputCatalogs"])
        all_manifest = self._concat_handles(inputs["inputManifests"])

        if len(all_catalog) > 0:
            circ = np.asarray(all_catalog["circularity"])
            finite = np.isfinite(circ)
            order = np.argsort(circ[finite])[::-1]
            finite_rows = all_catalog[finite]
            top_n = min(self.config.nRoundest, len(finite_rows))
            top_catalog = finite_rows[order[:top_n]]
        else:
            top_catalog = Table()

        # Attach a likely cutout path for convenience.
        if len(top_catalog) > 0 and len(all_manifest) > 0:
            all_cut = all_manifest[all_manifest["kind"] == "roundest_cutout"]
            if len(all_cut) > 0:
                by_fp = {int(r["footprint_id"]): str(r["path"]) for r in all_cut}
                top_catalog["roundest_cutout_path"] = [by_fp.get(int(fp), "") for fp in top_catalog["footprint_id"]]

        instrument = inputRefs.inputCatalogs[0].dataId["instrument"] if inputRefs.inputCatalogs else "unknown"
        out_dir = Path(self.config.outputRoot) / str(instrument)
        plot_rows = self._make_plots(all_catalog, out_dir)

        stats_manifest = Table(rows=plot_rows, names=["kind", "path"]) if plot_rows else Table(
            names=["kind", "path"], dtype=[str, str]
        )

        butlerQC.put(
            pipeBase.Struct(
                outputAllCatalog=all_catalog,
                outputTopCatalog=top_catalog,
                outputStatsManifest=stats_manifest,
            ),
            outputRefs,
        )
