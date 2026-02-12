# This file is part of transient_finder.

from __future__ import annotations

__all__ = [
    "ConsolidateDarkSourceCatalogsConnections",
    "ConsolidateDarkSourceCatalogConfig",
    "ConsolidateDarkSourceCatalogTask",
]

import lsst.pipe.base as pipeBase
from lsst.obs.base.utils import TableVStack
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.coaddBase import reorderRefs


class ConsolidateDarkSourceCatalogsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    inputCatalogs = connectionTypes.Input(
        doc="Input per-detector dark source catalogs.",
        name="detector_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Per-exposure concatenation of dark source catalogs.",
        name="exposure_dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
    )


class ConsolidateDarkSourceCatalogConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ConsolidateDarkSourceCatalogsConnections,
):
    pass


class ConsolidateDarkSourceCatalogTask(pipeBase.PipelineTask):
    _DefaultName = "consolidateDarkSourceCatalog"
    ConfigClass = ConsolidateDarkSourceCatalogConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.inputCatalogs]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")
        inputs = butlerQC.get(inputRefs)
        table = TableVStack.vstack_handles(inputs["inputCatalogs"])
        butlerQC.put(pipeBase.Struct(outputCatalog=table), outputRefs)
