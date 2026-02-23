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

import astropy.units as u
import numpy as np
from astropy.table import Table

import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.detection import Psf
from lsst.afw.image import Exposure
from lsst.meas.algorithms import SourceDetectionTask, Stamp, Stamps, SubtractBackgroundTask

__all__ = ["FindDarkSourcesTask", "FindDarkSourcesTaskConfig"]


class FindDarkSourcesConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "exposure", "detector")
):
    inputExp = cT.Input(
        name="post_isr_dark_CR_removed",
        doc="Input post isr dark with cosmic rays removed.",
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
        doc="Output combined proposed calibration.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )

    stampsOut = cT.Output(
        name="detector_dark_source_stamps",
        doc="Stamps of detected dark sources.",
        storageClass="Stamps",
        dimensions=("instrument", "exposure", "detector"),
    )


class FindDarkSourcesTaskConfig(pipeBase.PipelineTaskConfig, pipelineConnections=FindDarkSourcesConnections):
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

    def run(self, inputExp: Exposure, inputPsf: Psf) -> pipeBase.Struct:
        """Run source detection on darks and keep only
        the relevant round sources.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed dark frame data to combine.
        inputPsf : `lsst.afw.detection.Psf`
            Input PSF exposure.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``sourcesCatalog``
                Catalog of detected dark sources
                (`lsst.afw.table.SourceTable`).
            ``stampsOut``
                Stamps of detected dark sources
                (`lsst.meas.algorithms.Stamps`).
        """
        visit_info = inputExp.getInfo().getVisitInfo()
        exposure_id = visit_info.id
        detector_id = inputExp.getDetector().getId()
        self.log.info(f"Running FindDarkSourcesTask on visit {exposure_id}, detector {detector_id}")

        background = self.subtractBackgroundTask.run(exposure=inputExp)
        inputExp.setPsf(inputPsf)

        id_generator = measBase.IdGenerator()
        table = afwTable.SourceTable.make(self.schema, id_generator.make_table_id_factory())

        detections = self.sourceDetectionTask.run(
            table=table,
            exposure=inputExp,
            background=background.background,
        )
        areas = []
        fluxes = []
        t_vals = []
        e1_vals = []
        e2_vals = []
        e_vals = []
        axis_ratios = []
        centroids_x = []
        centroids_y = []
        exposure_ids = []
        detector_ids = []
        source_ids = []

        stamps = Stamps([])
        for src in detections.sources:
            fp = src.getFootprint()
            area = fp.getArea()
            source_id = src.getId()

            bbox = fp.getBBox()
            ext = bbox.getDimensions()

            w = ext.getX()
            h = ext.getY()

            # Aspect ratio (always >= 1)
            aspect = max(w / h, h / w)

            # Reject extremely elongated shapes
            if aspect > 1.25:
                continue

            # Build footprint mask
            subexp = inputExp[bbox]
            image_array = subexp.image.array.copy()

            flux = image_array.sum()
            if flux < 250000:
                continue

            # moments over footprint pixels
            fp_mask = fp.spans.asArray()
            m = self.second_moments(image_array, fp_mask, bkg="median")

            if m is None:
                continue

            if m["axis_ratio"] > 1.4 or m["T"] < 2 or m["T"] > 100:
                continue

            areas.append(area)
            fluxes.append(flux)
            t_vals.append(m["T"])
            e1_vals.append(m["e1"])
            e2_vals.append(m["e2"])
            e_vals.append(m["e"])
            axis_ratios.append(m["axis_ratio"])
            centroids_x.append(fp.getCentroid().getX())
            centroids_y.append(fp.getCentroid().getY())
            exposure_ids.append(int(exposure_id))
            detector_ids.append(int(detector_id))
            source_ids.append(source_id)

            stamp = Stamp(
                stamp_im=subexp,
                metadata={"source_id": source_id, "visit": int(exposure_id), "detector": int(detector_id)},
            )
            stamps.append(stamp)

        self.log.info(f"Found {len(areas)} dark sources in visit {exposure_id}, detector {detector_id}.")
        sources_table = Table(
            {
                "area": np.array(areas, dtype=float) * u.pixel**2,
                "flux": np.array(fluxes, dtype=float) * u.electron,
                "t": np.array(t_vals, dtype=float) * u.pixel,
                "e1": np.array(e1_vals, dtype=float),
                "e2": np.array(e2_vals, dtype=float),
                "e": np.array(e_vals, dtype=float),
                "centroid_x": np.array(centroids_x, dtype=float) * u.pixel,
                "centroid_y": np.array(centroids_y, dtype=float) * u.pixel,
                "exposure_id": np.array(exposure_ids, dtype=np.int64),
                "detector_id": np.array(detector_ids, dtype=np.int16),
                "source_id": np.array(source_ids, dtype=np.int64),
                "axis_ratio": np.array(axis_ratios, dtype=float),
            }
        )

        return pipeBase.Struct(sourcesCatalog=sources_table, stampsOut=stamps)

    def second_moments(
        self, img: np.ndarray, mask: np.ndarray, bkg: str = "median", eps: float = 1e-12
    ) -> dict | None:
        """Flux-weighted centroid + 2nd central moments over masked pixels."""
        Img = np.asarray(img, dtype=float)

        if bkg == "median":
            Img = Img - np.median(Img[mask])  # background from masked pixels
        elif bkg is None:
            pass
        else:
            raise ValueError("bkg must be 'median' or None")

        # weights only inside mask
        W = np.where(mask, Img, 0.0)

        # keep only positive weights (stabilizes)
        W = np.where(W > 0, W, 0.0)

        F = W.sum()
        if F <= eps:
            return None

        H, WW = W.shape
        y, x = np.mgrid[0:H, 0:WW]

        x0 = (W * x).sum() / F
        y0 = (W * y).sum() / F

        dx = x - x0
        dy = y - y0

        Ixx = (W * dx * dx).sum() / F
        Iyy = (W * dy * dy).sum() / F
        Ixy = (W * dx * dy).sum() / F

        T = Ixx + Iyy
        e1 = (Ixx - Iyy) / (T + eps)
        e2 = (2 * Ixy) / (T + eps)
        e = np.hypot(e1, e2)

        # eigenvalues -> axis ratio proxy (>=1)
        # lambda1 >= lambda2
        disc = np.sqrt(max((Ixx - Iyy) ** 2 + 4 * Ixy**2, 0.0))
        lam1 = 0.5 * (T + disc)
        lam2 = 0.5 * (T - disc)
        axis_ratio = np.sqrt((lam1 + eps) / (lam2 + eps))  # major/minor

        return dict(
            flux=F,
            x0=x0,
            y0=y0,
            Ixx=Ixx,
            Iyy=Iyy,
            Ixy=Ixy,
            T=T,
            e1=e1,
            e2=e2,
            e=e,
            lam1=lam1,
            lam2=lam2,
            axis_ratio=axis_ratio,
        )
