#!/usr/bin/python
# (C) British Crown Copyright 2018-2019 Met Office.
# All rights reserved.
#
# This file is part of TAF Monitor.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from bisect import bisect_right

import numpy as np
from taf_monitor.airfields_and_thresholds import define_thresholds


class cloud(object):

    """Class for checking cloud components of tafs against metars."""

    def __init__(self, airfield):

        _, self.cloud_thresholds = define_thresholds(
            rules=airfield["rules"], helicopter=airfield["helicopter"]
        )
        self.rules = airfield["rules"]

        if self.rules == "civil":
            self.cavok_groups = ["CAVOK", "FEW", "SCT"]
            self.significant_groups = ["BKN", "OVC"]
            self.significant_height = 14
        else:
            self.cavok_groups = ["CAVOK", "FEW"]
            self.significant_groups = ["SCT", "BKN", "OVC"]
            self.significant_height = 24

    def check(self, base_cloud, tempo_cloud, metar_cloud, first_base=[]):

        # Start with assumption of no cloud busts
        cloud_ok, bust_type = True, None

        all_taf_cloud = base_cloud + tempo_cloud

        some_cloud_groups = ["FEW", "SCT", "BKN", "OVC", "VV///"]

        # Check if CAVOK in metar
        if self.rules == "civil":
            if metar_cloud[0] == "CAVOK" and any(
                sig_cloud in base_item
                for base_item in base_cloud
                for sig_cloud in self.significant_groups
            ):
                # Start with assumption that TAF bust with increased cloud
                cloud_ok, bust_type = False, 'increase'
                # Check for significant cloud in tempo groups to save bust
                for item in self.cavok_groups:
                    if any(item in c_item for c_item in tempo_cloud):
                        cloud_ok, bust_type = True, None
                        break

            # Check if any significant cloud in metar, but TAF has CAVOK, with
            # any allowances.
            if any("CAVOK" in item for item in all_taf_cloud) and any(
                sig_cloud in metar_item
                for metar_item in metar_cloud
                for sig_cloud in self.significant_groups
            ):
                # Start with assumption that TAF bust with decreased cloud
                cloud_ok, bust_type = False, 'decrease'
                # Check for allowances in TAF
                for item in some_cloud_groups:
                    if any(item in c_item for c_item in all_taf_cloud):
                        cloud_ok, bust_type = True, None
                        break

        # For military TAFs we need to check if CB have been reported in a
        # METAR but not allowed for in the TAF.
        if self.rules == "military":
            if any("CB" in item for item in metar_cloud):
                if not any("CB" in item for item in all_taf_cloud):
                    cloud_ok, bust_type = False, 'CB'

        # Dealing with significant cloud heights.
        # Extract the significant clouds.
        metar_significant_cloud = []
        taf_significant_cloud = []
        base_significant_cloud = []
        tempo_significant_cloud = []
        first_base_sig_cloud = []

        # Identify BKN, OVC, or sky obscured amounts of cloud.
        for item in metar_cloud:
            if any(sig_cloud in item for sig_cloud in self.significant_groups):
                metar_significant_cloud.append(int(item[3:6]))
            if "VV" in item:
                metar_significant_cloud.append(int(0))
        for item in all_taf_cloud:
            if any(sig_cloud in item for sig_cloud in self.significant_groups):
                taf_significant_cloud.append(int(item[3:6]))
            if "VV" in item:
                taf_significant_cloud.append(int(0))
        for item in base_cloud:
            if any(sig_cloud in item for sig_cloud in self.significant_groups):
                base_significant_cloud.append(int(item[3:6]))
            if "VV" in item:
                base_significant_cloud.append(int(0))
        for item in tempo_cloud:
            if any(sig_cloud in item for sig_cloud in self.significant_groups):
                tempo_significant_cloud.append(int(item[3:6]))
            if "VV" in item:
                tempo_significant_cloud.append(int(0))
        for item in first_base:
            if any(sig_cloud in item for sig_cloud in self.significant_groups):
                first_base_sig_cloud.append(int(item[3:6]))
            if "VV" in item:
                first_base_sig_cloud.append(int(0))

        if self.rules == "military":
            if metar_cloud[0] == "CAVOK" and any(
                np.array(taf_significant_cloud) < self.significant_height
            ):
                cloud_ok, bust_type = False, 'increase'
                for item in self.cavok_groups:
                    if any(item in c_item for c_item in tempo_cloud):
                        cloud_ok, bust_type = True, None
                        break

            # Check if any significant cloud in metar, but TAF has CAVOK, with
            # any allowances.
            if any("CAVOK" in item for item in all_taf_cloud) and any(
                np.array(metar_significant_cloud) < self.significant_height
            ):
                cloud_ok, bust_type = False, 'decrease'
                for item in some_cloud_groups:
                    if any(item in c_item for c_item in all_taf_cloud):
                        cloud_ok, bust_type = True, None
                        break

        # Sort significant cloud in metar and tempo groups in height ascending
        # order.
        metar_significant_cloud = sorted(metar_significant_cloud)
        tempo_significant_cloud = sorted(tempo_significant_cloud)
        taf_significant_cloud = sorted(taf_significant_cloud)
        base_significant_cloud = sorted(base_significant_cloud)
        first_base_sig_cloud = sorted(first_base_sig_cloud)

        # Significant cloud below the significant height in metar, and no
        # corresponding group in the TAF
        if self.sig_low_cloud(metar_significant_cloud) and not self.sig_low_cloud(
            taf_significant_cloud
        ):
            cloud_ok, bust_type = False, 'decrease'

        # Check base condition specific rules (significant cloud below the
        # relevant height with nothing equivalent in metar).
        if self.sig_low_cloud(base_significant_cloud) and not self.sig_low_cloud(
            metar_significant_cloud
        ):
            cloud_ok, bust_type = False, 'increase'            # Allow for taf containing temporary improvement  group
            # (e.g. TEMPO SCT014, TEMPO CAVOK, TEMPO BKN016)
            if tempo_cloud:
                for item in self.cavok_groups:
                    if any(item in c_item for c_item in tempo_cloud):
                        cloud_ok, bust_type = True, None

                if tempo_significant_cloud:
                    if tempo_significant_cloud[0] > self.significant_height:
                        cloud_ok, bust_type = True, None

        # Checking for cloud group heights when significant low cloud is
        # present in TAF and METAR
        if self.sig_low_cloud(metar_significant_cloud) and self.sig_low_cloud(
            taf_significant_cloud
        ):
            [cloud_min, cloud_max] = self.lookup_clouds(taf_significant_cloud)

            # METAR cloud below lowest allowed height.
            if metar_significant_cloud[0] < cloud_min:
                cloud_ok, bust_type = False, 'decrease'

            # METAR cloud above highest allowed in TAF.
            if (
                cloud_max <= 15
                and metar_significant_cloud[0] >= cloud_max
                and base_significant_cloud
            ):
                cloud_ok, bust_type = False, 'increase'
                if tempo_cloud:
                    for item in self.cavok_groups:
                        if any(item in c_item for c_item in tempo_cloud):
                            cloud_ok, bust_type = True, None
                if tempo_significant_cloud:
                    if tempo_significant_cloud[0] > self.significant_height:
                        cloud_ok = True, None

        # Final check to allow intermediate groups during BECMG change
        if first_base:
            if first_base_sig_cloud:
                f_min, f_max = self.lookup_clouds(first_base_sig_cloud)
            else:
                f_min, f_max = self.lookup_clouds([50])
            if base_significant_cloud:
                b_min, b_max = self.lookup_clouds(base_significant_cloud)
            else:
                b_min, b_max = self.lookup_clouds([50])
            if metar_significant_cloud:
                m_cloud = metar_significant_cloud[0]
            else:
                m_cloud = 50
            m_min, m_max = self.lookup_clouds([m_cloud])

            if m_cloud >= min(f_min, b_min) and m_cloud <= max(f_max, b_max):
                cloud_ok, bust_type = True, None

        return cloud_ok, bust_type

    def sig_low_cloud(self, cloud_in):
        if cloud_in and cloud_in[0] <= self.significant_height:
            return True
        else:
            return False

    # Return cloud ranges.
    def lookup_clouds(self, taf_significant_cloud):
        taf_min = taf_significant_cloud[0]
        taf_max = taf_significant_cloud[-1]

        cloud_min = self.cloud_thresholds[
            bisect_right(self.cloud_thresholds, taf_min) - 1
        ]

        # Handle TAFs with an 050 height cloud group.
        if taf_max >= self.cloud_thresholds[-1]:
            cloud_max = self.cloud_thresholds[-1]
        else:
            cloud_max = self.cloud_thresholds[
                bisect_right(self.cloud_thresholds, taf_max)
            ]

        return cloud_min, cloud_max
