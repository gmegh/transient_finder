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
from astropy.table import QTable, vstack

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
    transientMatchedCatalogOut = connectionTypes.Output(
        doc="Transient Catalog with reprocessed matched stars",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="transient_matched_catalog",
        multiple=False,
    )
    transientUnmatchedCatalogOut = connectionTypes.Output(
        doc="Transient Catalog with reprocessed unmatched stars",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="transient_unmatched_catalog",
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

                assert main_visit_id in seen or main_visit_id in to_do, (
                    f"DataId {main_visit_id} not found in seen or to_do sets."
                )

                inputs = adjuster.get_inputs(data_id)
                adjuster.add_input(main_visit_id, "detectionCatalogs", inputs["detectionCatalogs"][0])
                adjuster.remove_quantum(data_id)

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

        # We need to ensure we always have the order of visits
        visits = [ref.dataId["visit"] for ref in inputRefs.detectionCatalogs]
        if visits[0] > visits[1]:
            detection_catalogs.reverse()
            visits.reverse()
        first_visit, second_visit = visits
        first_catalog, second_catalog = detection_catalogs
        self.log.info(f"Processing visits: {first_visit}, {second_visit}")

        first_catalog = first_catalog[~first_catalog["sky_source"]]
        second_catalog = second_catalog[~second_catalog["sky_source"]]
        self.log.info(f"First catalog size after sky_source cut: {len(first_catalog)}")
        self.log.info(f"Second catalog size after sky_source cut: {len(second_catalog)}")

        first_catalog = first_catalog[first_catalog["detect_isPrimary"]]
        second_catalog = second_catalog[second_catalog["detect_isPrimary"]]
        self.log.info(f"First catalog size after isPrimary cut: {len(first_catalog)}")
        self.log.info(f"Second catalog size after isPrimary cut: {len(second_catalog)}")

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
        ]

        def build_good_mask(cat: QTable) -> np.ndarray:
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

        matched_table = QTable(
            {
                "first_visit": np.full(len(matched1), first_visit, dtype="int64"),
                "second_visit": np.full(len(matched1), second_visit, dtype="int64"),
                "first_src_id": matched1["sourceId"],
                "second_src_id": matched2["sourceId"],
                "flux_diff12": flux_diff12 * flux_unit,
                "flux_diff09": flux_diff09 * flux_unit,
                "flux_diff06": flux_diff06 * flux_unit,
                "flux_diffpsf": flux_diffpsf * flux_unit,
                "extendedness": extendedness,
            }
        )
        self.log.info("Matched table constructed")

        first_unmatched_table = QTable(
            {
                "visit": np.full(len(unmatched_prev), first_visit, dtype="int64"),
                "other_visit": np.full(len(unmatched_prev), second_visit, dtype="int64"),
                "sourceId": unmatched_prev["sourceId"],
                "ra": unmatched_prev["coord_ra"] * u.deg,
                "dec": unmatched_prev["coord_dec"] * u.deg,
                "ap12Flux": unmatched_prev["ap12Flux"] * u.nJy,
                "extendedness": unmatched_prev["sizeExtendedness"],
            }
        )
        self.log.info("First unmatched table constructed")

        second_unmatched_table = QTable(
            {
                "visit": np.full(len(unmatched_curr), second_visit, dtype="int64"),
                "other_visit": np.full(len(unmatched_curr), first_visit, dtype="int64"),
                "sourceId": unmatched_curr["sourceId"],
                "ra": unmatched_curr["coord_ra"] * u.deg,
                "dec": unmatched_curr["coord_dec"] * u.deg,
                "ap12Flux": unmatched_curr["ap12Flux"] * u.nJy,
                "extendedness": unmatched_curr["sizeExtendedness"],
            }
        )
        self.log.info("Second unmatched table constructed")
        unmatched_table = vstack([first_unmatched_table, second_unmatched_table])
        self.log.info("Unmatched table constructed")

        butlerQC.put(unmatched_table, outputRefs.transientUnmatchedCatalogOut)
        butlerQC.put(matched_table, outputRefs.transientMatchedCatalogOut)
