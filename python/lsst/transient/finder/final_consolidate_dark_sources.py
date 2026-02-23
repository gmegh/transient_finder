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
    "FinalConsolidateDarkSourcesConnections",
    "FinalConsolidateDarkSourcesConfig",
    "FinalConsolidateDarkSourcesTask",
]


import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import Stamps
from lsst.obs.base.utils import TableVStack
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.coaddBase import reorderRefs


class FinalConsolidateDarkSourcesConnections(
    pipeBase.PipelineTaskConnections, defaultTemplates={"catalogType": ""}, dimensions=("instrument",)
):
    inputCatalogs = connectionTypes.Input(
        doc="Input per-detector dark source catalogs",
        name="dark_source_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    inputStamps = connectionTypes.Input(
        doc="Input per-detector dark source stamps",
        name="dark_source_stamps",
        storageClass="Stamps",
        dimensions=("instrument", "exposure"),
        multiple=True,
        deferLoad=True,
    )
    outputCatalog = connectionTypes.Output(
        doc="Per-instrument concatenation of dark source catalogs",
        name="dark_source_catalog_final",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
    )
    outputStamps = connectionTypes.Output(
        doc="Per-instrument concatenation of dark source stamps",
        name="dark_source_stamps_final",
        storageClass="Stamps",
        dimensions=("instrument",),
    )


class FinalConsolidateDarkSourcesConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=FinalConsolidateDarkSourcesConnections
):
    pass


class FinalConsolidateDarkSourcesTask(pipeBase.PipelineTask):
    """Concatenate `detector_dark_source_catalog`
    list into a per-instrument `dark_source_catalog`
    """

    _DefaultName = "finalConsolidateDarkSources"
    ConfigClass = FinalConsolidateDarkSourcesConfig

    inputDataset = "dark_source_catalog"
    outputDataset = "dark_source_catalog_final"

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        # Docstring inherited.
        detectorOrderCatalogs = [ref.dataId["exposure"] for ref in inputRefs.inputCatalogs]
        detectorOrderCatalogs.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrderCatalogs, dataIdKey="exposure")
        inputs = butlerQC.get(inputRefs)
        self.log.info("Concatenating %s per-exposure dark source catalogs", len(inputs["inputCatalogs"]))
        table = TableVStack.vstack_handles(inputs["inputCatalogs"])
        butlerQC.put(pipeBase.Struct(outputCatalog=table), outputRefs)

        detectorOrderStamps = [ref.dataId["exposure"] for ref in inputRefs.inputStamps]
        detectorOrderStamps.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrderStamps, dataIdKey="exposure")
        inputs = butlerQC.get(inputRefs)
        self.log.info("Concatenating %s per-exposure dark source stamps", len(inputs["inputStamps"]))
        stamps = Stamps([])
        for handle in inputs["inputStamps"]:
            stamp_list = handle.get()
            print(stamp_list)
            if len(stamp_list) > 0:
                stamps.extend(stamp_list)
        butlerQC.put(pipeBase.Struct(outputStamps=stamps), outputRefs)
