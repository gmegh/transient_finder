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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ["FindDarkSourcesTaskConnections", "FindDarkSourcesTaskConfig", "FindDarkSourcesTask"]

import numpy as np

import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.detection import Psf
from lsst.afw.image import Exposure
from lsst.meas.algorithms import SourceDetectionTask, SubtractBackgroundTask


class FindDarkSourcesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure", "detector"),
):
    inputExp = cT.Input(
        name="post_isr_dark",
        doc="Input post-ISR dark exposure.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )

    inputPsf = cT.Input(
        name="input_psf_exposure",
        doc="Input PSF exposure.",
        storageClass="Psf",
        dimensions=("instrument",),
    )

    sourcesCatalog = cT.Output(
        name="detector_dark_source_catalog",
        doc="Detected dark sources with roundness and size metrics.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )


class FindDarkSourcesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FindDarkSourcesTaskConnections,
):
    sourceDetectionTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Task for source detection.",
    )

    subtractBackgroundTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task for background subtraction.",
    )

    minRoundness = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Minimum roundness to keep in output catalog. 1.0 is perfectly round.",
    )

    minDiameterPixels = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc="Minimum equivalent footprint diameter (pixels) to keep in output catalog.",
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.sourceDetectionTask.doTempLocalBackground = False
        self.sourceDetectionTask.doTempWideBackground = False


class FindDarkSourcesTask(pipeBase.PipelineTask):
    """Detect dark-frame sources and compute roundness and size metrics."""

    ConfigClass = FindDarkSourcesTaskConfig
    _DefaultName = "findDarkSources"

    def __init__(self, schema: afwTable.Schema | None = None, **kwargs):
        super().__init__(**kwargs)

        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()

        self.makeSubtask("subtractBackgroundTask")
        self.makeSubtask("sourceDetectionTask")

        schema.addField("exposure", type="L", doc="Exposure this source appeared on.")
        schema.addField("detector", type="I", doc="Detector this source appeared on.")
        schema.addField("centroid_x", type="D", doc="Footprint centroid x in pixels.")
        schema.addField("centroid_y", type="D", doc="Footprint centroid y in pixels.")
        schema.addField("footprint_area", type="D", doc="Footprint area in pixels.")
        schema.addField("equivalent_diameter", type="D", doc="Equivalent circular diameter from area in pixels.")
        schema.addField("footprint_i_xx", type="D", doc="Footprint Ixx second moment in pixels^2.")
        schema.addField("footprint_i_yy", type="D", doc="Footprint Iyy second moment in pixels^2.")
        schema.addField("footprint_i_xy", type="D", doc="Footprint Ixy second moment in pixels^2.")
        schema.addField("footprint_ellipticity", type="D", doc="Ellipticity derived from footprint moments.")
        schema.addField("roundness", type="D", doc="Roundness proxy 1 - ellipticity.")

        self.schema = schema

    def _compute_shape_metrics(
        self, source: afwTable.SourceRecord
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        footprint = source.getFootprint()
        center = footprint.getCentroid()
        area = float(footprint.getArea())
        equivalent_diameter = float(np.sqrt((4.0 * area) / np.pi)) if area > 0 else np.nan

        shape = footprint.getShape()
        i_xx = float(shape.getIxx())
        i_yy = float(shape.getIyy())
        i_xy = float(shape.getIxy())
        denom = i_xx + i_yy
        if denom <= 0.0:
            return (
                float(center.getX()),
                float(center.getY()),
                area,
                equivalent_diameter,
                i_xx,
                i_yy,
                i_xy,
                np.nan,
                np.nan,
            )

        ellipticity = float(np.sqrt((i_xx - i_yy) ** 2 + 4.0 * i_xy**2) / denom)
        roundness = float(np.clip(1.0 - ellipticity, 0.0, 1.0))
        return (
            float(center.getX()),
            float(center.getY()),
            area,
            equivalent_diameter,
            i_xx,
            i_yy,
            i_xy,
            ellipticity,
            roundness,
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):  # type: ignore[no-untyped-def]
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(
            inputExp=inputs["inputExp"],
            inputPsf=inputs["inputPsf"],
            exposure=inputRefs.inputExp.dataId["exposure"],
            detector=inputRefs.inputExp.dataId["detector"],
        )
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp: Exposure, inputPsf: Psf, exposure: int, detector: int) -> pipeBase.Struct:
        background = self.subtractBackgroundTask.run(exposure=inputExp)
        inputExp.setPsf(inputPsf)

        id_generator = measBase.IdGenerator()
        table = afwTable.SourceTable.make(self.schema, id_generator.make_table_id_factory())

        detections = self.sourceDetectionTask.run(
            table=table,
            exposure=inputExp,
            background=background.background,
        )
        sources = detections.sources

        for source in sources:
            source["exposure"] = exposure
            source["detector"] = detector
            (
                centroid_x,
                centroid_y,
                area,
                equivalent_diameter,
                i_xx,
                i_yy,
                i_xy,
                ellipticity,
                roundness,
            ) = self._compute_shape_metrics(source)
            source["centroid_x"] = centroid_x
            source["centroid_y"] = centroid_y
            source["footprint_area"] = area
            source["equivalent_diameter"] = equivalent_diameter
            source["footprint_i_xx"] = i_xx
            source["footprint_i_yy"] = i_yy
            source["footprint_i_xy"] = i_xy
            source["footprint_ellipticity"] = ellipticity
            source["roundness"] = roundness

        output_table = sources.asAstropy()

        if self.config.minRoundness > 0.0:
            output_table = output_table[output_table["roundness"] >= self.config.minRoundness]
        if self.config.minDiameterPixels > 0.0:
            output_table = output_table[output_table["equivalent_diameter"] >= self.config.minDiameterPixels]

        return pipeBase.Struct(sourcesCatalog=output_table)
