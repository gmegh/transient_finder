# This file is part of transient_finder.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "ConsolidateTransientCatalogsConnections",
    "ConsolidateTransientCatalogConfig",
    "ConsolidateTransientCatalogTask",
]


import lsst.pipe.base as pipeBase
from lsst.obs.base.utils import TableVStack
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.coaddBase import reorderRefs


class ConsolidateTransientCatalogsConnections(
    pipeBase.PipelineTaskConnections, defaultTemplates={"catalogType": ""}, dimensions=("instrument", "visit")
):
    inputCatalogs = connectionTypes.Input(
        doc="Input per-detector transient unmatched catalogs",
        name="detector_transient_unmatched_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Per-visit concatenation of transient catalogs",
        name="transient_unmatched_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )


class ConsolidateTransientCatalogConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ConsolidateTransientCatalogsConnections
):
    pass


class ConsolidateTransientCatalogTask(pipeBase.PipelineTask):
    """Concatenate `detector_transient_unmatched_catalog`
    list into a per-visit `transient_unmatched_catalog`
    """

    _DefaultName = "consolidateTransientCatalog"
    ConfigClass = ConsolidateTransientCatalogConfig

    inputDataset = "detector_transient_unmatched_catalog"
    outputDataset = "transient_unmatched_catalog"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        # Docstring inherited.
        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.inputCatalogs]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")
        inputs = butlerQC.get(inputRefs)
        self.log.info(
            "Concatenating %s per-detector transient unmatched catalogs", len(inputs["inputCatalogs"])
        )
        table = TableVStack.vstack_handles(inputs["inputCatalogs"])
        butlerQC.put(pipeBase.Struct(outputCatalog=table), outputRefs)
