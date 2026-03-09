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
    "DifferenceImagingTaskConnections",
    "DifferenceImagingTaskConfig",
    "DifferenceImagingTask",
]


from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import Table

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.geom import makeWcsPairTransform
from lsst.daf.butler import DataCoordinate
from lsst.ip.diffim.detectAndMeasure import DetectAndMeasureTask
from lsst.ip.diffim.subtractImages import AlardLuptonSubtractTask
from lsst.meas.algorithms import Stamp, Stamps, WarpedPsf
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.measurementDriver import ForcedMeasurementDriverTask

bad_flags = [
    "sky_source",
    "base_CircularApertureFlux_flag_badCentroid",
    "base_PsfFlux_flag_badCentroid",
    "ext_shapeHSM_HsmPsfMoments_flag_badCentroid",
    "ext_shapeHSM_HsmSourceMoments_flag_badCentroid",
    "ip_diffim_DipoleFit_classification",
    "ip_diffim_DipoleFit_flag",
    "base_PixelFlags_flag",
    "base_PixelFlags_flag_offimage",
    "base_PixelFlags_flag_edge",
    "base_PixelFlags_flag_nodata",
    "base_PixelFlags_flag_interpolated",
    "base_PixelFlags_flag_saturated",
    "base_PixelFlags_flag_cr",
    "base_PixelFlags_flag_bad",
    "base_PixelFlags_flag_suspect",
    "base_PixelFlags_flag_edgeCenter",
    "base_PixelFlags_flag_nodataCenter",
    "base_PixelFlags_flag_interpolatedCenter",
    "base_PixelFlags_flag_saturatedCenter",
    "base_PixelFlags_flag_crCenter",
    "base_PixelFlags_flag_badCenter",
    "base_PixelFlags_flag_suspectCenter",
    "base_PixelFlags_flag_edgeCenterAll",
    "base_PixelFlags_flag_nodataCenterAll",
    "base_PixelFlags_flag_interpolatedCenterAll",
    "base_PixelFlags_flag_saturatedCenterAll",
    "base_PixelFlags_flag_crCenterAll",
    "base_PixelFlags_flag_badCenterAll",
    "base_PixelFlags_flag_suspectCenterAll",
    "base_PixelFlags_flag_streakCenter",
    "base_PixelFlags_flag_injectedCenter",
    "base_PixelFlags_flag_injected_templateCenter",
    "base_PixelFlags_flag_streakCenterAll",
    "base_PixelFlags_flag_injectedCenterAll",
    "base_PixelFlags_flag_injected_templateCenterAll",
    "base_PixelFlags_flag_streak",
    "base_PixelFlags_flag_injected",
    "base_PixelFlags_flag_injected_template",
]

FORBIDDEN_MASK_PLANES = {
    "SAT",
    "SPIKE",
    "UNMASKEDNAN",
    "STREAK",
    "CROSSTALK",
    "ITL_DIP",
    "CR",
    "BAD",
    "NO_DATA",
    "EDGE",
    "SUSPECT",
    "INTRP",
    "NOT_DEBLENDED",
}


class DifferenceImagingTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument", "detector"),  # type: ignore
):
    visitImages = connectionTypes.Input(
        doc="Preliminary Visit Images",
        dimensions=("visit", "instrument", "detector"),
        storageClass="ExposureF",
        name="preliminary_visit_image",
        multiple=True,
    )
    starFootprints = connectionTypes.Input(
        doc="Footprints of stars to create the matching kernel on",
        dimensions=("visit", "instrument", "detector"),
        storageClass="SourceCatalog",
        name="single_visit_star_footprints",
        multiple=False,
    )
    diffimCatalogOut = connectionTypes.Output(
        doc="Transient Catalog with reprocessed unmatched stars",
        dimensions=("visit", "instrument", "detector"),
        storageClass="ArrowAstropy",
        name="detector_diffim_sources",
        multiple=False,
    )
    diffimStampsOut = connectionTypes.Output(
        name="detector_diffim_stamps",
        doc="Stamps of detected dark sources.",
        storageClass="Stamps",
        dimensions=("visit", "instrument", "detector"),
    )
    firstVisitStampsOut = connectionTypes.Output(
        name="detector_first_visit_stamps",
        doc="Stamps of detected dark sources.",
        storageClass="Stamps",
        dimensions=("visit", "instrument", "detector"),
    )
    secondVisitStampsOut = connectionTypes.Output(
        name="detector_second_visit_stamps",
        doc="Stamps of detected dark sources.",
        storageClass="Stamps",
        dimensions=("visit", "instrument", "detector"),
    )

    def adjust_all_quanta(self, adjuster: pipeBase.QuantaAdjuster) -> None:
        """This will drop intra quanta and assign
        them to the extra detector quanta
        """
        consecutive_pair_table = adjuster.butler.get("consecutive_exposure_pairs")
        to_do = set(adjuster.iter_data_ids())
        seen = set()
        while to_do:
            data_id = to_do.pop()
            if data_id["visit"] in consecutive_pair_table["visit_id"]:
                seen.add(data_id)
            elif data_id["visit"] in consecutive_pair_table["prev_visit_id"]:
                row = consecutive_pair_table[consecutive_pair_table["prev_visit_id"] == data_id["visit"]]
                main_visit_id = DataCoordinate.standardize(data_id, visit=row["visit_id"].value[0])

                if main_visit_id not in seen and main_visit_id not in to_do:
                    adjuster.remove_quantum(data_id)
                    continue

                inputs = adjuster.get_inputs(data_id)
                adjuster.add_input(main_visit_id, "visitImages", inputs["visitImages"][0])
                adjuster.remove_quantum(data_id)

            else:
                adjuster.remove_quantum(data_id)


class DifferenceImagingTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DifferenceImagingTaskConnections,  # type: ignore
):
    subtractTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=AlardLuptonSubtractTask, doc="Task for subtracting."
    )
    detectTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=DetectAndMeasureTask, doc="Task for detecting."
    )
    forcedPhotometryTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=ForcedMeasurementDriverTask, doc="Task for WCS fitting."
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.subtractTask.mode = "auto"
        self.forcedPhotometryTask.measurement.slots.psfFlux = "base_PsfFlux"
        self.forcedPhotometryTask.measurement.slots.centroid = "base_TransformedCentroidFromCoord"
        self.forcedPhotometryTask.measurement.slots.shape = None
        self.forcedPhotometryTask.measurement.doReplaceWithNoise = False
        self.forcedPhotometryTask.doApCorr = True


class DifferenceImagingTask(pipeBase.PipelineTask):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = DifferenceImagingTaskConfig
    _DefaultName = "DifferenceImagingTask"
    config: DifferenceImagingTaskConfig
    subtractTask: AlardLuptonSubtractTask
    detectTask: DetectAndMeasureTask
    forcedPhotometryTask: ForcedMeasurementDriverTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.makeSubtask("subtractTask")
        self.makeSubtask("detectTask")
        self.makeSubtask("forcedPhotometryTask")

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        self.log.info("Running DifferenceImagingTask")
        visit_images = butlerQC.get(inputRefs.visitImages)
        direct_sources = butlerQC.get(inputRefs.starFootprints)

        visits = [ref.dataId["visit"] for ref in inputRefs.visitImages]
        if len(visits) != 2:
            self.log.warning(f"Missing one of the visits, number of visits found: {len(visits)}")
            return
        if visits[0] > visits[1]:
            visit_images.reverse()
            visits.reverse()
        first_visit, second_visit = visits
        first_visit_image, second_visit_image = visit_images
        detector_id = first_visit_image.getDetector().getId()
        band = first_visit_image.getFilter().bandLabel

        warpConfig = afwMath.Warper.ConfigClass()
        warpConfig.cacheSize = 100000
        warpConfig.interpLength = 100
        warpConfig.warpingKernelName = "lanczos3"
        warper = afwMath.Warper.fromConfig(warpConfig)

        bbox = first_visit_image.getBBox()
        final_bbox = lsst.geom.Box2I(bbox.getBegin(), bbox.getDimensions())
        final_bbox.grow(10)

        if (
            second_visit_image.wcs is None
            or first_visit_image.wcs is None
            or second_visit_image.getPsf() is None
            or first_visit_image.getPsf() is None
        ):
            self.log.warning("Missing WCS or PSF in one of the visit images, skipping difference imaging.")
            return

        transform = makeWcsPairTransform(second_visit_image.wcs, first_visit_image.wcs)
        warp_psf = WarpedPsf(second_visit_image.getPsf(), transform)
        warped_second_visit_image = warper.warpExposure(
            first_visit_image.wcs, second_visit_image, destBBox=final_bbox
        )
        warped_second_visit_image.setPsf(warp_psf)

        diffim_results = self.subtractTask.run(warped_second_visit_image, first_visit_image, direct_sources)
        detectResults = self.detectTask.run(
            first_visit_image, diffim_results.matchedTemplate, diffim_results.difference
        )

        preliminary_source_catalog_per_detector = detectResults.diaSources.asAstropy()
        diffim = detectResults.subtractedMeasuredExposure

        self.log.info("Building good masks for preliminary source catalog")
        first_good_masks = build_good_mask(preliminary_source_catalog_per_detector, first_visit_image)
        second_good_masks = build_good_mask(preliminary_source_catalog_per_detector, second_visit_image)
        both_good = first_good_masks & second_good_masks
        self.log.info(f"Number of sources with good masks in both visits: {np.sum(both_good)}")
        source_catalog_per_detector = preliminary_source_catalog_per_detector[both_good]

        self.log.info(
            "Running forced photometry on both visits at the positions of"
            "the diffim preliminary source catalog"
        )
        try:
            first_forced_photometry = self.forcedPhotometryTask.runFromAstropy(
                source_catalog_per_detector,
                first_visit_image,
                id_column_name="sourceId",
                ra_column_name="ra",
                dec_column_name="dec",
                psf_footprint_scaling=3.0,
            )

            second_forced_photometry = self.forcedPhotometryTask.runFromAstropy(
                source_catalog_per_detector,
                second_visit_image,
                id_column_name="sourceId",
                ra_column_name="ra",
                dec_column_name="dec",
                psf_footprint_scaling=3.0,
            )
        except RuntimeError as e:
            if "without a PSF" in str(e):
                return
            raise

        # sign = +1 where `visit` is the first visit, -1 where it's the second
        sign = np.where(source_catalog_per_detector["is_negative"].data, -1, 1)
        self.log.info("Adding metadata to source catalog")
        source_catalog_per_detector["first_visit"] = np.full(
            len(source_catalog_per_detector), first_visit, dtype="int64"
        )
        source_catalog_per_detector["second_visit"] = np.full(
            len(source_catalog_per_detector), second_visit, dtype="int64"
        )
        source_catalog_per_detector["detector"] = np.full(
            len(source_catalog_per_detector), detector_id, dtype="int64"
        )
        source_catalog_per_detector["band"] = np.full(len(source_catalog_per_detector), band, dtype="str")
        source_catalog_per_detector["x"] = source_catalog_per_detector["slot_Centroid_x"]
        source_catalog_per_detector["y"] = source_catalog_per_detector["slot_Centroid_y"]

        self.log.info("Filtering forced photometry catalogs")
        # Forced photometry fluxes on each visit
        source_catalog_per_detector["forced_psfFlux_first_visit"] = (
            first_forced_photometry["base_PsfFlux_instFlux"].data * u.electron
        )
        source_catalog_per_detector["forced_psfFlux_second_visit"] = (
            second_forced_photometry["base_PsfFlux_instFlux"].data * u.electron
        )

        source_catalog_per_detector["forced_ap03Flux_first_visit"] = (
            first_forced_photometry["base_CircularApertureFlux_3_0_instFlux"].data * u.electron
        )
        source_catalog_per_detector["forced_ap03Flux_second_visit"] = (
            second_forced_photometry["base_CircularApertureFlux_3_0_instFlux"].data * u.electron
        )

        # Forced differences
        source_catalog_per_detector["forced_psfFlux_diff"] = (
            sign
            * (
                first_forced_photometry["base_PsfFlux_instFlux"].data
                - second_forced_photometry["base_PsfFlux_instFlux"].data
            )
            * u.electron
        )

        source_catalog_per_detector["forced_ap03Flux_diff"] = (
            sign
            * (
                first_forced_photometry["base_CircularApertureFlux_3_0_instFlux"].data
                - second_forced_photometry["base_CircularApertureFlux_3_0_instFlux"].data
            )
            * u.electron
        )

        source_catalog_per_detector["forced_ap06Flux_diff"] = (
            sign
            * (
                first_forced_photometry["base_CircularApertureFlux_6_0_instFlux"].data
                - second_forced_photometry["base_CircularApertureFlux_6_0_instFlux"].data
            )
            * u.electron
        )

        source_catalog_per_detector["forced_ap09Flux_diff"] = (
            sign
            * (
                first_forced_photometry["base_CircularApertureFlux_9_0_instFlux"].data
                - second_forced_photometry["base_CircularApertureFlux_9_0_instFlux"].data
            )
            * u.electron
        )

        source_catalog_per_detector["forced_ap12Flux_diff"] = (
            sign
            * (
                first_forced_photometry["base_CircularApertureFlux_12_0_instFlux"].data
                - second_forced_photometry["base_CircularApertureFlux_12_0_instFlux"].data
            )
            * u.electron
        )

        source_catalog_per_detector["forced_ap17Flux_diff"] = (
            sign
            * (
                first_forced_photometry["base_CircularApertureFlux_17_0_instFlux"].data
                - second_forced_photometry["base_CircularApertureFlux_17_0_instFlux"].data
            )
            * u.electron
        )

        # Minimum with measured source-catalog fluxes
        source_catalog_per_detector["ap12Flux_min"] = (
            np.minimum(
                source_catalog_per_detector["forced_ap12Flux_diff"].to_value(u.electron),
                source_catalog_per_detector["base_CircularApertureFlux_12_0_instFlux"],
            )
            * u.electron
        )

        source_catalog_per_detector["psfFlux_min"] = (
            np.minimum(
                source_catalog_per_detector["forced_psfFlux_diff"].to_value(u.electron),
                source_catalog_per_detector["base_PsfFlux_instFlux"],
            )
            * u.electron
        )

        diffim_stamps = Stamps([])
        first_visit_stamps = Stamps([])
        second_visit_stamps = Stamps([])
        for source in detectResults.diaSources:
            source_id = source.getId()
            detector_id = source.get("detector")
            fp = source.getFootprint()
            bbox = fp.getBBox()

            diffim_sub_image = diffim[bbox]
            first_sub_image = first_visit_image[bbox]
            second_sub_image = second_visit_image[bbox]
            diffim_stamps.append(
                Stamp(
                    stamp_im=diffim_sub_image,
                    metadata={
                        "source_id": source_id,
                        "first_visit": int(first_visit),
                        "second_visit": int(second_visit),
                        "detector": int(detector_id),
                    },
                )
            )
            first_visit_stamps.append(
                Stamp(
                    stamp_im=first_sub_image,
                    metadata={
                        "source_id": source_id,
                        "first_visit": int(first_visit),
                        "second_visit": int(second_visit),
                        "detector": int(detector_id),
                    },
                )
            )
            second_visit_stamps.append(
                Stamp(
                    stamp_im=second_sub_image,
                    metadata={
                        "source_id": source_id,
                        "first_visit": int(first_visit),
                        "second_visit": int(second_visit),
                        "detector": int(detector_id),
                    },
                )
            )

        butlerQC.put(source_catalog_per_detector, outputRefs.diffimCatalogOut)
        butlerQC.put(diffim_stamps, outputRefs.diffimStampsOut)
        butlerQC.put(first_visit_stamps, outputRefs.firstVisitStampsOut)
        butlerQC.put(second_visit_stamps, outputRefs.secondVisitStampsOut)


def _mask_plane_good(exposure: afwImage.Exposure, x: float, y: float) -> bool:
    """Return True if the mask at (x, y) only has allowed planes set."""
    mask = exposure.mask
    planes = mask.interpret(mask[x, y])  # list of plane names -> set

    tokens = {p.strip() for p in planes.split(",")}
    return tokens.isdisjoint(FORBIDDEN_MASK_PLANES)


def build_good_mask(cat: Table, exposure: afwImage.Exposure) -> np.ndarray:
    """Combine catalog bad_flags and exposure mask planes."""
    m = np.ones(len(cat), dtype=bool)

    for flag in bad_flags:
        m &= ~cat[flag]

    # 2) image mask-plane check at the source position
    # use the centroid in the forced phot table (change names if yours differ)
    x = cat["slot_Centroid_x"]
    y = cat["slot_Centroid_y"]

    mask_plane_ok = np.array(
        [_mask_plane_good(exposure, xi, yi) for xi, yi in zip(x, y)],
        dtype=bool,
    )
    m &= mask_plane_ok
    return m
