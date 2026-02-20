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
    "FilterTransientsTaskConnections",
    "FilterTransientsTaskConfig",
    "FilterTransientsTask",
]


from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import Table

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butler import DataCoordinate
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks.measurementDriver import (
    ForcedMeasurementDriverTask,
)

bad_flags = [
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


class FilterTransientsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument", "detector"),  # type: ignore
):
    transientCatalogs = connectionTypes.Input(
        doc="Transient Catalog",
        dimensions=("visit", "instrument"),
        storageClass="ArrowAstropy",
        name="preliminary_transient_unmatched_catalog",
        multiple=False,
    )
    visitImages = connectionTypes.Input(
        doc="Preliminary Visit Images",
        dimensions=("visit", "instrument", "detector"),
        storageClass="ExposureF",
        name="preliminary_visit_image",
        multiple=True,
    )
    detectorTransientUnmatchedCatalogOut = connectionTypes.Output(
        doc="Transient Catalog with reprocessed unmatched stars",
        dimensions=("visit", "instrument", "detector"),
        storageClass="ArrowAstropy",
        name="detector_transient_unmatched_catalog",
        multiple=False,
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


class FilterTransientsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FilterTransientsTaskConnections,  # type: ignore
):
    forcedPhotometryTask: pexConfig.ConfigurableField = pexConfig.ConfigurableField(
        target=ForcedMeasurementDriverTask, doc="Task for WCS fitting."
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.forcedPhotometryTask.measurement.slots.psfFlux = "base_PsfFlux"
        self.forcedPhotometryTask.measurement.slots.centroid = "base_TransformedCentroidFromCoord"
        self.forcedPhotometryTask.measurement.slots.shape = None
        self.forcedPhotometryTask.measurement.doReplaceWithNoise = False
        self.forcedPhotometryTask.doApCorr = True


class FilterTransientsTask(pipeBase.PipelineTask):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = FilterTransientsTaskConfig
    _DefaultName = "FilterTransientsTask"
    config: FilterTransientsTaskConfig
    forcedPhotometryTask: ForcedMeasurementDriverTask

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.makeSubtask("forcedPhotometryTask")

    def detectors_missing_wcs(self, det_table: Table) -> set[int]:
        return {rec["id"] for rec in det_table if rec.wcs is None}

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        self.log.info("Running FilterTransientTask")
        visit_images = butlerQC.get(inputRefs.visitImages)
        transient_catalog = butlerQC.get(inputRefs.transientCatalogs)

        visits = [ref.dataId["visit"] for ref in inputRefs.visitImages]
        if len(visits) != 2:
            self.log.warning(f"Missing one of the visits, number of visits found: {len(visits)}")
            return
        if visits[0] > visits[1]:
            visit_images.reverse()
            visits.reverse()
        first_visit, second_visit = visits
        first_visit_image, second_visit_image = visit_images

        transient_catalog_per_detector = transient_catalog[
            transient_catalog["detector"] == inputRefs.visitImages[0].dataId["detector"]
        ]

        try:
            first_forced_photometry = self.forcedPhotometryTask.runFromAstropy(
                transient_catalog_per_detector,
                first_visit_image,
                id_column_name="sourceId",
                ra_column_name="ra",
                dec_column_name="dec",
                psf_footprint_scaling=3.0,
            )

            second_forced_photometry = self.forcedPhotometryTask.runFromAstropy(
                transient_catalog_per_detector,
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

        self.log.info("Building good masks for both catalogs")
        good_first = build_good_mask(first_forced_photometry, first_visit_image)
        good_second = build_good_mask(second_forced_photometry, second_visit_image)
        both_good = good_first & good_second

        visit = transient_catalog_per_detector["visit"].data

        # sign = +1 where `visit` is the first visit, -1 where it's the second
        sign = np.where(
            visit == first_visit,
            1.0,
            np.where(visit == second_visit, -1.0, np.nan),
        )

        # sanity check: any rows not matching either visit?
        if np.any(np.isnan(sign)):
            self.log.warning(
                "Found rows with 'visit' neither first_visit (%s) nor second_visit (%s).",
                first_visit,
                second_visit,
            )

        self.log.info("Filtering forced photometry catalogs")
        first_forced_photometry = first_forced_photometry[both_good]
        second_forced_photometry = second_forced_photometry[both_good]
        transient_catalog_per_detector = transient_catalog_per_detector[both_good]
        sign = sign[both_good]

        forced_unmatched_table = Table(
            {
                "visit": transient_catalog_per_detector["visit"],
                "other_visit": transient_catalog_per_detector["other_visit"],
                "day_obs": transient_catalog_per_detector["day_obs"],
                "sourceId": transient_catalog_per_detector["sourceId"],
                "ra": transient_catalog_per_detector["ra"] * u.deg,
                "dec": transient_catalog_per_detector["dec"] * u.deg,
                "x": transient_catalog_per_detector["x"],
                "y": transient_catalog_per_detector["y"],
                "detector": transient_catalog_per_detector["detector"],
                "band": transient_catalog_per_detector["band"],
                "ap03Flux": transient_catalog_per_detector["ap03Flux"],
                "ap06Flux": transient_catalog_per_detector["ap06Flux"],
                "ap09Flux": transient_catalog_per_detector["ap09Flux"],
                "ap12Flux": transient_catalog_per_detector["ap12Flux"],
                "ap17Flux": transient_catalog_per_detector["ap17Flux"],
                "psfFlux": transient_catalog_per_detector["psfFlux"],
                "sky": transient_catalog_per_detector["sky"],
                "extendedness": transient_catalog_per_detector["extendedness"],
                "ixx": transient_catalog_per_detector["ixx"],
                "iyy": transient_catalog_per_detector["iyy"],
                "ixy": transient_catalog_per_detector["ixy"],
                # instrument fluxes
                "ap12_instFlux": transient_catalog_per_detector["ap12_instFlux"],
                "ap17_instFlux": transient_catalog_per_detector["ap17_instFlux"],
                "ap35_instFlux": transient_catalog_per_detector["ap35_instFlux"],
                "ap50_instFlux": transient_catalog_per_detector["ap50_instFlux"],
                "tophat_instFlux": transient_catalog_per_detector["tophat_instFlux"],
                "localBackground_instFlux": transient_catalog_per_detector["localBackground_instFlux"],
                # forced photometry fluxes
                # in electrons because I ran it on preliminary_visit_image
                "forced_psfFlux_first_visit": first_forced_photometry["base_PsfFlux_instFlux"].data
                * u.electron,
                "forced_psfFlux_second_visit": second_forced_photometry["base_PsfFlux_instFlux"].data
                * u.electron,
                "forced_ap03Flux_first_visit": first_forced_photometry[
                    "base_CircularApertureFlux_3_0_instFlux"
                ].data
                * u.electron,
                "forced_ap03Flux_second_visit": second_forced_photometry[
                    "base_CircularApertureFlux_3_0_instFlux"
                ].data
                * u.electron,
                "forced_psfFlux_diff": sign
                * (
                    first_forced_photometry["base_PsfFlux_instFlux"].data
                    - second_forced_photometry["base_PsfFlux_instFlux"].data
                )
                * u.electron,
                "forced_ap03Flux_diff": sign
                * (
                    first_forced_photometry["base_CircularApertureFlux_3_0_instFlux"].data
                    - second_forced_photometry["base_CircularApertureFlux_3_0_instFlux"].data
                )
                * u.electron,
                "forced_ap06Flux_diff": sign
                * (
                    first_forced_photometry["base_CircularApertureFlux_6_0_instFlux"].data
                    - second_forced_photometry["base_CircularApertureFlux_6_0_instFlux"].data
                )
                * u.electron,
                "forced_ap09Flux_diff": sign
                * (
                    first_forced_photometry["base_CircularApertureFlux_9_0_instFlux"].data
                    - second_forced_photometry["base_CircularApertureFlux_9_0_instFlux"].data
                )
                * u.electron,
                "forced_ap12Flux_diff": sign
                * (
                    first_forced_photometry["base_CircularApertureFlux_12_0_instFlux"].data
                    - second_forced_photometry["base_CircularApertureFlux_12_0_instFlux"].data
                )
                * u.electron,
                "forced_ap17Flux_diff": sign
                * (
                    first_forced_photometry["base_CircularApertureFlux_17_0_instFlux"].data
                    - second_forced_photometry["base_CircularApertureFlux_17_0_instFlux"].data
                )
                * u.electron,
            }
        )
        forced_unmatched_table["ap12Flux_min"] = (
            np.minimum(
                forced_unmatched_table["forced_ap12Flux_diff"].to(u.electron).value,
                forced_unmatched_table["ap12_instFlux"].to(u.electron).value,
            )
            * u.electron
        )
        # TO-DO this is incorrect can't do diff of electron to nJy
        forced_unmatched_table["ap17Flux_min"] = (
            np.minimum(
                forced_unmatched_table["forced_ap17Flux_diff"].to(u.electron).value,
                forced_unmatched_table["ap17Flux"].to(u.nJy).value,
            )
            * u.nJy
        )
        butlerQC.put(forced_unmatched_table, outputRefs.detectorTransientUnmatchedCatalogOut)


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
