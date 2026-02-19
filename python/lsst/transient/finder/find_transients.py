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
    "FindTransientsTaskConnections",
    "FindTransientsTaskConfig",
    "FindTransientsTask",
]

from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

import lsst.pipe.base as pipeBase
from lsst.daf.butler import DataCoordinate
from lsst.pipe.base import connectionTypes


class FindTransientsTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("visit", "instrument"),  # type: ignore
):
    detectionCatalogs = connectionTypes.Input(
        doc="Single Visit Stars Reprocessed Catalogs",
        dimensions=("visit", "instrument"),
        storageClass="ArrowAstropy",
        name="single_visit_star_reprocessed",
        multiple=True,
    )
    visitSummaries = connectionTypes.Input(
        doc="Preliminary Visit Summary Catalogs",
        dimensions=("visit", "instrument"),
        storageClass="ExposureCatalog",
        name="preliminary_visit_summary",
        multiple=True,
    )
    transientMatchedCatalogOut = connectionTypes.Output(
        doc="Transient Catalog with reprocessed matched stars",
        dimensions=("visit", "instrument"),
        storageClass="ArrowAstropy",
        name="preliminary_transient_matched_catalog",
        multiple=False,
    )
    transientUnmatchedCatalogOut = connectionTypes.Output(
        doc="Transient Catalog with reprocessed unmatched stars",
        dimensions=("visit", "instrument"),
        storageClass="ArrowAstropy",
        name="preliminary_transient_unmatched_catalog",
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
                row = consecutive_pair_table[consecutive_pair_table["visit_id"] == data_id["visit"]]
                main_visit_id = DataCoordinate.standardize(data_id, visit=row["prev_visit_id"].value[0])

                if main_visit_id not in seen and main_visit_id not in to_do:
                    adjuster.remove_quantum(data_id)
                    continue

                inputs = adjuster.get_inputs(data_id)
                adjuster.add_input(main_visit_id, "detectionCatalogs", inputs["detectionCatalogs"][0])
                adjuster.add_input(main_visit_id, "visitSummaries", inputs["visitSummaries"][0])
            elif data_id["visit"] in consecutive_pair_table["prev_visit_id"]:
                seen.add(data_id)
                row = consecutive_pair_table[consecutive_pair_table["prev_visit_id"] == data_id["visit"]]
                main_visit_id = DataCoordinate.standardize(data_id, visit=row["visit_id"].value[0])

                if main_visit_id not in seen and main_visit_id not in to_do:
                    adjuster.remove_quantum(data_id)
                    continue

                inputs = adjuster.get_inputs(data_id)
                adjuster.add_input(main_visit_id, "detectionCatalogs", inputs["detectionCatalogs"][0])
                adjuster.add_input(main_visit_id, "visitSummaries", inputs["visitSummaries"][0])
            else:
                adjuster.remove_quantum(data_id)


class FindTransientsTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FindTransientsTaskConnections,  # type: ignore
):
    pass


class FindTransientsTask(pipeBase.PipelineTask):
    """
    Cut out the donut postage stamps on corner wavefront sensors (CWFS)
    """

    ConfigClass = FindTransientsTaskConfig
    _DefaultName = "FindTransientsTask"
    config: FindTransientsTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def detectors_missing_wcs(self, det_table: Table) -> set[int]:
        return {rec["id"] for rec in det_table if rec.wcs is None}

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        self.log.info("Running FindTransientsTask")
        flux_unit = u.nJy
        radius = 1.0 * u.arcsec
        detection_catalogs = butlerQC.get(inputRefs.detectionCatalogs)
        visit_summaries = butlerQC.get(inputRefs.visitSummaries)

        # We need to ensure we always have the order of visits
        visits = [ref.dataId["visit"] for ref in inputRefs.detectionCatalogs]
        if len(visits) != 2:
            self.log.warning(f"Missing one of the visits, number of visits found: {len(visits)}")
            return
        if visits[0] > visits[1]:
            detection_catalogs.reverse()
            visit_summaries.reverse()
            visits.reverse()

        first_visit, second_visit = visits
        first_catalog, second_catalog = detection_catalogs
        first_summary, second_summary = visit_summaries
        self.log.info(f"Processing visits: {first_visit}, {second_visit}")

        first_catalog = first_catalog[~first_catalog["sky_source"]]
        second_catalog = second_catalog[~second_catalog["sky_source"]]
        self.log.info(f"First catalog size after sky_source cut: {len(first_catalog)}")
        self.log.info(f"Second catalog size after sky_source cut: {len(second_catalog)}")

        first_catalog = first_catalog[first_catalog["detect_isPrimary"]]
        second_catalog = second_catalog[second_catalog["detect_isPrimary"]]
        self.log.info(f"First catalog size after isPrimary cut: {len(first_catalog)}")
        self.log.info(f"Second catalog size after isPrimary cut: {len(second_catalog)}")

        # these are chosen by hand based on how vignetted they are.
        # TO-DO: make this configurable
        bad_detectors = {1, 75, 117}
        first_wcs_missing_detectors = self.detectors_missing_wcs(first_summary)
        second_wcs_missing_detectors = self.detectors_missing_wcs(second_summary)
        wcs_missing_detectors = first_wcs_missing_detectors | second_wcs_missing_detectors
        skipped_detectors = list(bad_detectors | wcs_missing_detectors)
        self.log.info(f"Skipping detectors: {skipped_detectors}")
        first_detectors_mask = ~np.isin(first_catalog["detector"], skipped_detectors)
        second_detectors_mask = ~np.isin(second_catalog["detector"], skipped_detectors)
        first_catalog = first_catalog[first_detectors_mask]
        second_catalog = second_catalog[second_detectors_mask]
        self.log.info(
            f"First catalog size after detector cut: {len(first_catalog)}, "
            f"dropped {(~first_detectors_mask).sum()} sources"
        )
        self.log.info(
            f"Second catalog size after detector cut: {len(second_catalog)}, "
            f"dropped {(~second_detectors_mask).sum()} sources"
        )

        # Match sky coordinates
        first_coords = SkyCoord(first_catalog["coord_ra"], first_catalog["coord_dec"], unit="deg")
        second_coords = SkyCoord(second_catalog["coord_ra"], second_catalog["coord_dec"], unit="deg")
        idx, sep2d, _ = first_coords.match_to_catalog_sky(second_coords)
        mask_match = sep2d < radius
        self.log.info(f"Number of matched sources: {np.sum(mask_match)}")

        # --- Unmatched stars ---
        bad_flags = [
            "pixelFlags_bad",
            "pixelFlags_cr",
            "pixelFlags_crCenter",
            "pixelFlags_edge",
            "pixelFlags_interpolated",
            "pixelFlags_interpolatedCenter",
            "pixelFlags_nodata",
            "pixelFlags_offimage",
            "pixelFlags_saturated",
            "pixelFlags_saturatedCenter",
            "pixelFlags_suspect",
            "pixelFlags_suspectCenter",
            "deblend_skipped",
            "centroid_flag",
            "invalidPsfFlag",
        ]

        def build_good_mask(cat: Table) -> np.ndarray:
            m = np.ones(len(cat), dtype=bool)
            for flag in bad_flags:
                m &= ~cat[flag]
            return m

        self.log.info("Building good masks for both catalogs")
        good_prev = build_good_mask(first_catalog)
        good_curr = build_good_mask(second_catalog)
        self.log.info(f"Number of good sources in first catalog: {np.sum(good_prev)}")
        self.log.info(f"Number of good sources in second catalog: {np.sum(good_curr)}")

        # --- Matched stars (both matched AND both good) ---
        matched1 = first_catalog[mask_match]
        matched2 = second_catalog[idx[mask_match]]

        good_pair = good_prev[mask_match] & good_curr[idx[mask_match]]

        matched1 = matched1[good_pair]
        matched2 = matched2[good_pair]
        self.log.info(f"Number of matched and good sources: {len(matched1)}")

        # --- Unmatched stars (unmatched AND good) ---
        self.log.info("Building unmatched catalogs")
        unmatched_prev = first_catalog[(~mask_match) & good_prev]

        matched_curr_idx = idx[mask_match]
        is_matched_curr = np.zeros(len(second_catalog), dtype=bool)
        is_matched_curr[matched_curr_idx] = True

        unmatched_curr = second_catalog[(~is_matched_curr) & good_curr]
        self.log.info(f"Number of unmatched sources in first catalog: {len(unmatched_prev)}")
        self.log.info(f"Number of unmatched sources in second catalog: {len(unmatched_curr)}")

        # compute scalar arrays only
        flux_diff12 = matched2["ap12Flux"] - matched1["ap12Flux"]
        flux_diff09 = matched2["ap09Flux"] - matched1["ap09Flux"]
        flux_diff06 = matched2["ap06Flux"] - matched1["ap06Flux"]
        flux_diffpsf = matched2["psfFlux"] - matched1["psfFlux"]
        extendedness = np.maximum(matched2["sizeExtendedness"], matched1["sizeExtendedness"])
        self.log.info("Computed flux differences and extendedness")

        day_obs_first = first_visit // 10**4
        day_obs_second = second_visit // 10**4

        # --- units ---
        flux_unit = u.nJy  # or whatever you're using for calibFlux
        inst_flux_unit = u.electron

        # --- instrumental fluxes (arrays) ---
        # visit 1
        ap12_1 = matched1["apFlux_12_0_instFlux"]
        ap17_1 = matched1["apFlux_17_0_instFlux"]
        ap35_1 = matched1["apFlux_35_0_instFlux"]
        ap50_1 = matched1["apFlux_50_0_instFlux"]
        tophat_1 = matched1["normCompTophatFlux_instFlux"]
        bkg_1 = matched1["localBackground_instFlux"]

        # visit 2
        ap12_2 = matched2["apFlux_12_0_instFlux"]
        ap17_2 = matched2["apFlux_17_0_instFlux"]
        ap35_2 = matched2["apFlux_35_0_instFlux"]
        ap50_2 = matched2["apFlux_50_0_instFlux"]
        tophat_2 = matched2["normCompTophatFlux_instFlux"]
        bkg_2 = matched2["localBackground_instFlux"]

        # differences (2 - 1)
        dap12 = ap12_2 - ap12_1
        dap17 = ap17_2 - ap17_1
        dap35 = ap35_2 - ap35_1
        dap50 = ap50_2 - ap50_1
        dtophat = tophat_2 - tophat_1
        dbkg = bkg_2 - bkg_1

        matched_table = Table(
            {
                # visits / ids
                "first_visit": np.full(len(matched1), first_visit, dtype="int64"),
                "second_visit": np.full(len(matched1), second_visit, dtype="int64"),
                "day_obs": np.full(len(matched1), day_obs_first, dtype="int64"),
                "first_src_id": matched1["sourceId"],
                "second_src_id": matched2["sourceId"],
                # positions (use coord_* as you had)
                "ra": matched1["coord_ra"] * u.deg,
                "dec": matched1["coord_dec"] * u.deg,
                "x": matched1["x"],
                "y": matched1["y"],
                "detector": matched1["detector"],
                "band": matched1["band"],
                # shape moments (pixels^2)
                "ixx": matched1["ixx"] * u.pixel**2,
                "iyy": matched1["iyy"] * u.pixel**2,
                "ixy": matched1["ixy"] * u.pixel**2,
                # calibrated flux differences
                "flux_diff12": flux_diff12 * flux_unit,
                "flux_diff09": flux_diff09 * flux_unit,
                "flux_diff06": flux_diff06 * flux_unit,
                "flux_diffpsf": flux_diffpsf * flux_unit,
                # instrumental fluxes: visit 1
                "ap12_instFlux_1": ap12_1 * inst_flux_unit,
                "ap17_instFlux_1": ap17_1 * inst_flux_unit,
                "ap35_instFlux_1": ap35_1 * inst_flux_unit,
                "ap50_instFlux_1": ap50_1 * inst_flux_unit,
                "tophat_instFlux_1": tophat_1 * inst_flux_unit,
                "localBackground_instFlux_1": bkg_1 * inst_flux_unit,
                # instrumental fluxes: visit 2
                "ap12_instFlux_2": ap12_2 * inst_flux_unit,
                "ap17_instFlux_2": ap17_2 * inst_flux_unit,
                "ap35_instFlux_2": ap35_2 * inst_flux_unit,
                "ap50_instFlux_2": ap50_2 * inst_flux_unit,
                "tophat_instFlux_2": tophat_2 * inst_flux_unit,
                "localBackground_instFlux_2": bkg_2 * inst_flux_unit,
                # instrumental flux differences (2 - 1)
                "ap12_instFlux_diff": dap12 * inst_flux_unit,
                "ap17_instFlux_diff": dap17 * inst_flux_unit,
                "ap35_instFlux_diff": dap35 * inst_flux_unit,
                "ap50_instFlux_diff": dap50 * inst_flux_unit,
                "tophat_instFlux_diff": dtophat * inst_flux_unit,
                "localBackground_instFlux_diff": dbkg * inst_flux_unit,
                # morphology classifier
                "extendedness": extendedness,
            }
        )
        self.log.info("Matched table constructed")

        # ---------- first_unmatched_table (previous visit) ----------
        first_unmatched_table = Table(
            {
                "visit": np.full(len(unmatched_prev), first_visit, dtype="int64"),
                "other_visit": np.full(len(unmatched_prev), second_visit, dtype="int64"),
                "day_obs": np.full(len(unmatched_prev), day_obs_first, dtype="int64"),
                "sourceId": unmatched_prev["sourceId"],
                "ra": unmatched_prev["coord_ra"] * u.deg,
                "dec": unmatched_prev["coord_dec"] * u.deg,
                "x": unmatched_prev["x"],
                "y": unmatched_prev["y"],
                "detector": unmatched_prev["detector"],
                "band": unmatched_prev["band"],
                # shape moments
                "ixx": unmatched_prev["ixx"] * u.pixel**2,
                "iyy": unmatched_prev["iyy"] * u.pixel**2,
                "ixy": unmatched_prev["ixy"] * u.pixel**2,
                # calibrated fluxes
                "ap03Flux": unmatched_prev["ap03Flux"] * flux_unit,
                "ap06Flux": unmatched_prev["ap06Flux"] * flux_unit,
                "ap09Flux": unmatched_prev["ap09Flux"] * flux_unit,
                "ap12Flux": unmatched_prev["ap12Flux"] * flux_unit,
                "ap17Flux": unmatched_prev["ap17Flux"] * flux_unit,
                "psfFlux": unmatched_prev["psfFlux"] * flux_unit,
                # instrumental fluxes (electrons)
                "ap12_instFlux": unmatched_prev["apFlux_12_0_instFlux"] * inst_flux_unit,
                "ap17_instFlux": unmatched_prev["apFlux_17_0_instFlux"] * inst_flux_unit,
                "ap35_instFlux": unmatched_prev["apFlux_35_0_instFlux"] * inst_flux_unit,
                "ap50_instFlux": unmatched_prev["apFlux_50_0_instFlux"] * inst_flux_unit,
                "tophat_instFlux": unmatched_prev["normCompTophatFlux_instFlux"] * inst_flux_unit,
                "localBackground_instFlux": unmatched_prev["localBackground_instFlux"] * inst_flux_unit,
                # background / morphology
                "sky": unmatched_prev["sky"],
                "extendedness": unmatched_prev["sizeExtendedness"],
            }
        )
        self.log.info("First unmatched table constructed")

        second_unmatched_table = Table(
            {
                "visit": np.full(len(unmatched_curr), second_visit, dtype="int64"),
                "other_visit": np.full(len(unmatched_curr), first_visit, dtype="int64"),
                "day_obs": np.full(len(unmatched_curr), day_obs_second, dtype="int64"),
                "sourceId": unmatched_curr["sourceId"],
                "ra": unmatched_curr["coord_ra"] * u.deg,
                "dec": unmatched_curr["coord_dec"] * u.deg,
                "x": unmatched_curr["x"],
                "y": unmatched_curr["y"],
                "detector": unmatched_curr["detector"],
                "band": unmatched_curr["band"],
                # shape moments
                "ixx": unmatched_curr["ixx"] * u.pixel**2,
                "iyy": unmatched_curr["iyy"] * u.pixel**2,
                "ixy": unmatched_curr["ixy"] * u.pixel**2,
                # calibrated fluxes
                "ap03Flux": unmatched_curr["ap03Flux"] * flux_unit,
                "ap06Flux": unmatched_curr["ap06Flux"] * flux_unit,
                "ap09Flux": unmatched_curr["ap09Flux"] * flux_unit,
                "ap12Flux": unmatched_curr["ap12Flux"] * flux_unit,
                "ap17Flux": unmatched_curr["ap17Flux"] * flux_unit,
                "psfFlux": unmatched_curr["psfFlux"] * flux_unit,
                # instrumental fluxes (electrons)
                "ap12_instFlux": unmatched_curr["apFlux_12_0_instFlux"] * inst_flux_unit,
                "ap17_instFlux": unmatched_curr["apFlux_17_0_instFlux"] * inst_flux_unit,
                "ap35_instFlux": unmatched_curr["apFlux_35_0_instFlux"] * inst_flux_unit,
                "ap50_instFlux": unmatched_curr["apFlux_50_0_instFlux"] * inst_flux_unit,
                "tophat_instFlux": unmatched_curr["normCompTophatFlux_instFlux"] * inst_flux_unit,
                "localBackground_instFlux": unmatched_curr["localBackground_instFlux"] * inst_flux_unit,
                # background / morphology
                "sky": unmatched_curr["sky"],
                "extendedness": unmatched_curr["sizeExtendedness"],
            }
        )
        self.log.info("Second unmatched table constructed")
        unmatched_table = vstack([first_unmatched_table, second_unmatched_table])
        self.log.info("Unmatched table constructed")

        butlerQC.put(unmatched_table, outputRefs.transientUnmatchedCatalogOut)
        butlerQC.put(matched_table, outputRefs.transientMatchedCatalogOut)
