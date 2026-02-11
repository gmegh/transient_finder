# This file is part of transient_finder.
#
# Developed for the LSST Data Management System.
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import math

import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import Exposure
from lsst.meas.algorithms import SourceDetectionTask, SubtractBackgroundTask

__all__ = ["FindDarkSourcesTask", "FindDarkSourcesTaskConfig"]


class FindDarkSourcesConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "exposure", "detector")
):
    inputExp = cT.Input(
        name="post_isr_dark",
        doc="Input post isr dark.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )

    sourcesCatalog = cT.Output(
        name="detector_dark_source_catalog",
        doc="Output combined proposed calibration.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )


class FindDarkSourcesTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=FindDarkSourcesConnections):
    psfFwhm: pexConfig.Field = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc="Repair PSF FWHM (pixels).",
    )
    psfSize: pexConfig.Field = pexConfig.Field(
        dtype=int,
        default=21,
        doc="Repair PSF size (pixels).",
    )

    sourceDetectionTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SourceDetectionTask, doc="Task for source detection."
    )

    subtractBackgroundTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask, doc="Task for background subtraction."
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.sourceDetectionTask.doTempLocalBackground = False
        self.sourceDetectionTask.doTempWideBackground = False


class FindDarkSourcesTask(pipeBase.PipelineTask):
    """Combine pre-processed dark frames into a proposed master calibration."""

    ConfigClass = FindDarkSourcesTaskConfig
    _DefaultName = "findDarkSources"
    config: FindDarkSourcesTaskConfig
    subtractBackgroundTask: SubtractBackgroundTask
    sourceDetectionTask: SourceDetectionTask

    def __init__(self, schema=None, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)

        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("subtractBackgroundTask")
        self.makeSubtask("sourceDetectionTask")

        schema.addField(
            "visit",
            type="L",
            doc="Visit this source appeared on.",
        )
        schema.addField(
            "detector",
            type="U",
            doc="Detector this source appeared on.",
        )

        self.schema = schema

    def run(self, inputExp: Exposure) -> pipeBase.Struct:
        """Preprocess input exposures prior to DARK combination.

        This task detects and repairs cosmic rays strikes.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed dark frame data to combine.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputExp``
                CR rejected, ISR processed Dark Frame
                (`lsst.afw.image.Exposure`).
        """
        background = self.subtractBackgroundTask.run(exposure=inputExp)

        psf = measAlg.SingleGaussianPsf(
            self.config.psfSize, self.config.psfSize, self.config.psfFwhm / (2 * math.sqrt(2 * math.log(2)))
        )
        inputExp.setPsf(psf)

        id_generator = measBase.IdGenerator()
        table = afwTable.SourceTable.make(self.schema, id_generator.make_table_id_factory())

        detections = self.sourceDetectionTask.run(
            table=table,
            exposure=inputExp,
            background=background,
        )
        sources = detections.sources

        return pipeBase.Struct(
            sourcesCatalog=sources,
        )
