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

from taf_monitor.constants import *


class weather(object):
    def __init__(self, airfield):
        pass

    def check(self, base_weather, tempo_weather, metar_weather,
              first_base=[]):

        all_taf_weather = base_weather + tempo_weather

        # Check base and TEMPOs together
        weather_ok = self.wx_check(metar_weather, all_taf_weather,
                                   base_weather)

        return weather_ok, None

    def wx_check(self, metar_weather, taf_wx, base_weather, ch_tempo=False):

        weather_ok = True

        if metar_weather:
            for item in metar_weather:
                for item_s in special_types:
                    if item_s in item and not any(
                        item_s in w_item for w_item in taf_wx
                    ):
                        weather_ok = False
                        # if not "VC" in item: weather_ok = False

        # Check to ensure a taf visibility does not get mistaken for a
        # precipitation group. Counts no. of taf weather groups that are
        # vis type and determines if they all are by comparison with length
        # of taf_wx list. If they all are vis and the metar weather
        # is not all vis type a precip group is needed in the taf.
        # This test also covers cases in which the metar contains weather,
        # but taf contains no weather at all as the first test, length of
        # vis types compared to taf_wx length will give 0=0.
        if len(set(taf_wx).intersection(vis_types)) == len(
            taf_wx
        ) and not len(set(metar_weather).intersection(vis_types)) == len(metar_weather):
            weather_ok = False

        if not ch_tempo:
            # Check for non-visibility, non-light weather in base of taf, but only
            # visibility weather type or none in metar.
            # Uses similar length comparisons as above, but with slightly altered
            # logic. Allows for NSW as weather type.
            if not (
                len(set(base_weather).intersection(vis_types)) == len(base_weather)
            ) and (len(set(metar_weather).intersection(vis_types)) == len(metar_weather)):

                weather_ok = False
                for item in taf_wx:
                    if "NSW" in item:
                        weather_ok = True

        # To cope with NSW need to check that if taf contains only vis type
        # groups and an NSW group and the metars give precip, then the taf is
        # bust.
        if len(set(taf_wx).intersection(vis_types)) == len(taf_wx
        ) - 1 and not len(set(metar_weather).intersection(vis_types)) == len(
            metar_weather
        ):

            for item in taf_wx:
                # Find the extra group to be NSW rather than precip, stop
                # checking and return False (broken taf).
                if item == "NSW":
                    return False

        return weather_ok
