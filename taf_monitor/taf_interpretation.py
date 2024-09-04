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

import inspect

import taf_monitor.weather_regex as wr
from taf_monitor.time_functionality import ConstructTimeObject, Is_Time_Current
from taf_monitor.utilities import DefineConditionsDictionary, error_routine


class Simplify_TAF(object):
    ######################################
    #                                    #
    #        SIMPLIFY TAF/METAR          #
    #                                    #
    ######################################

    def by_group(self, TAF):
        """
        Make PROB## TEMPO groups and PROB## groups all into TEMPO groups.

        """
        for idx, item in enumerate(TAF):
            if item == "TEMPO" and "PROB" in TAF[idx - 1]:
                TAF.pop(idx - 1)
        for idx, item in enumerate(TAF):
            if "PROB" in item:
                TAF[idx] = "TEMPO"
        return TAF

    def get_valid_tempo_groups(
        self,
        TAF,
        indices,
        metar_time,
        change_groups,
        tempo_changes,
        condition_type,
        icao,
        current_day,
        current_month,
        current_year,
    ):
        """
        Filter tempo groups for those valid at metar time.

        """
        each_tempo = DefineConditionsDictionary().create(condition_type)
        for item in indices:
            [start_time, end_time] = ConstructTimeObject(
                TAF[item + 1], current_day, current_month, current_year
            ).TAF()

            if Is_Time_Current(start_time, end_time, metar_time).check():
                each_tempo = Interpret_TAF().conditions(
                    TAF[item : change_groups[change_groups.index(item) + 1]],
                    tempo_changes,
                    icao,
                )
                tempo_changes.update(each_tempo)

        return tempo_changes

    def kill_elements(self, TAF, kill_list):
        """
        Kill off elements of TAF that are not current at time of metar.

        """
        TAF = [item for i, item in enumerate(TAF) if i not in kill_list]
        return TAF

    def apply_becmg(self, base_conditions, becmg_changes, condition_type):
        """
        Apply changes from completed becmg groups into the taf base conditions.

        """

        for item in condition_type:
            if becmg_changes[item]:
                # Use last element for a given type (with [-1]) incase two or
                # more becmg groups have completed for the same weather element
                base_conditions[item] = [becmg_changes[item][-1]]

        return base_conditions


class Interpret_TAF(object):
    ######################################
    #                                    #
    #         EXTRACT CONDITIONS         #
    #                                    #
    ######################################

    def conditions(self, list_in, list_out, icao):
        """
        Extract TAF elements from simplified input and break into categories.
        """
        for item in list_in:
            if not any([item == test for test in [icao, "TEMPO", "BECMG"]]):
                if wr.PRESSURE_TERM.match(item):
                    break
                if wr.WIND_TERM.match(item) is not None:
                    list_out["wind"].append(item)
                if wr.VISIBILITY_TERM.match(item) is not None:
                    list_out["visibility"].append(item)
                    # Do not search for weather until visibility has been
                    # found.
                    weather_search = True
                if item == "CAVOK":
                    list_out["visibility"].append("9999")
                    list_out["cloud"].append("CAVOK")
                    weather_search = True
                    # Avoid CAVOK being caught in the weather group by
                    # advancing to next item.
                    continue

                #                if item == "NCD" and list_out["visibility"][-1] == '9999':
                #                    list_out["cloud"].append('CAVOK')
                # Avoid NCD being caught in the weather group by advancing
                # to next item.
                #                    continue

                if (
                    item == "NCD"
                    or item == "NSC"
                    or item == "//////CB"
                    or item == "//////TCU"
                    or "///" in item
                ):
                    list_out["cloud"].append("FEW049")
                    # Avoid NSC being caught in the weather group by advancing
                    # to next item.
                    continue
                if wr.WEATHER_TERM.search(item) is not None:
                    # Throw away vicinity weather phenomena.
                    if "VC" not in item:
                        list_out["weather"].append(item)
                if wr.CLOUD_TERM.match(item) is not None:
                    list_out["cloud"].append(item)
                    # Breaks to avoid runway states and recent weather
                    # conditions.
        return list_out

    def get_indices(self, TAF, search_string):
        """
        Search TAF for given search_string and return indices of matches.

        """
        return [item[0] for item in enumerate(TAF) if search_string in item]

    def get_all_indices(self, TAF):
        """
        Identify indices of given types of groups.
        """
        # Obtain indices of BECMG groups.
        becmg_groups = self.get_indices(TAF, "BECMG")
        # Obtain indices of TEMPO groups.
        tempo_groups = self.get_indices(TAF, "TEMPO")
        # All change groups.
        change_groups = sorted(becmg_groups + tempo_groups)
        # Return position of last element in taf.
        change_groups.append(len(TAF))

        return becmg_groups, tempo_groups, change_groups

    def interpret_becmg_group(
        self,
        TAF,
        indices,
        metar_time,
        change_groups,
        becmg_changes,
        becmg_period,
        icao,
        current_day,
        current_month,
        current_year,
    ):
        """
        Find current and completed becmg groups, throw away becmg groups yet to
        become active.

        """
        kill_list = []
        for item in indices:
            [start_time, end_time] = ConstructTimeObject(
                TAF[item + 1], current_day, current_month, current_year
            ).TAF()
            # becmg group finished, so modify base conditions.
            if end_time <= metar_time:
                becmg_changes = Interpret_TAF().conditions(
                    TAF[item : change_groups[change_groups.index(item) + 1]],
                    becmg_changes,
                    icao,
                )
            # becmg group change still in progress, so make it into a tempo
            # group which is equivalent.
            elif Is_Time_Current(start_time, end_time, metar_time).check():

                # Add to becmg_period group
                becmg_period = Interpret_TAF().conditions(
                    TAF[item : change_groups[change_groups.index(item) + 1]],
                    becmg_period,
                    icao,
                )

                # Have to treat wind group becmg slightly differently as moving
                # from 180->330 converted to a tempo group would leave 250 and
                # 260 degrees as invalid, despite it being a becmg.
                # for item_w in TAF[item : change_groups[change_groups.index(item) + 1]]:
                #     if wr.WIND_TERM.match(item_w):
                #         TAF[item + 2] = str("BEC" + TAF[item + 2][3:])
                # TAF[item] = "TEMPO"



            # becmg group yet to begin, so discard it.
            elif start_time > metar_time:
                try:
                    last_index = change_groups[change_groups.index(item) + 1]
                except IndexError:
                    # TODO Changed above from bare except to IndexError.
                    # If problems, be aware of this.
                    error_routine(inspect.stack(), "Error interpreting BECMG group")
                    last_index = len(TAF)
                kill_list = kill_list + list(range(item, last_index))

        return TAF, kill_list, becmg_changes, becmg_period
