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

import platform
from datetime import datetime as dt
from datetime import timedelta as timedelta

from taf_monitor.airfields_and_thresholds import define_airfields, define_benches
from taf_monitor.cloud_check import cloud
from taf_monitor.constants import condition_type
from taf_monitor.data_retrieval import RetrieveObservations
from taf_monitor.taf_interpretation import Interpret_TAF, Simplify_TAF
from taf_monitor.time_functionality import ConstructTimeObject, Is_Time_Current
from taf_monitor.utilities import DefineConditionsDictionary
from taf_monitor.visibility_check import visibility
from taf_monitor.weather_check import weather
from taf_monitor.wind_check import wind

# Create dict type to contain airfield lists.
airfield_benches = define_benches()
all_airfields = define_airfields()


class CheckTafThread():

    def __init__(self, icao, taf_start_time, taf_end_time, taf, metars):
        self.icao = icao
        self.taf_start_time = taf_start_time
        self.taf_end_time = taf_end_time
        self.taf = taf
        self.metars = metars

    def time_in_seconds(self, td):
        return (
            td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6
        ) / 10 ** 6

    def check_condition(self, function, condition, airfield):
        """
        Call the relevant checking plugin for a given condition (e.g. wind,
        cloud, etc). The taf_status is then updated relative to this condition.
        """
        taf_status_1 = self.taf_status[condition].copy()
        taf_status_2 = self.taf_status[condition].copy()

        (taf_status_1, bust_types) = function(airfield).check(
            self.base_conditions[condition],
            self.tempo_changes[condition],
            self.metar_conditions[condition],
        )

        if self.base_conditions_2[condition]:
            (taf_status_2, bust_types_2) = function(airfield).check(
                self.base_conditions_2[condition],
                self.tempo_changes[condition],
                self.metar_conditions[condition],
                first_base=self.base_conditions[condition]
            )

            self.taf_status[condition] = taf_status_1 or taf_status_2
            if not self.taf_status[condition]:

                # Get combined bust types
                if function == wind:
                    bust_types = {
                        b_type: bust_types[b_type] or bust_types_2[b_type]
                        for b_type in bust_types
                        }
                elif bust_types != bust_types_2:
                    if not bust_types or not bust_types_2:
                        bust_types = bust_types or bust_types_2
                    else:
                        bust_types = 'both'

            # Set bust types to None if TAF not bust
            else:
                bust_types = None

        else:
            self.taf_status[condition] = taf_status_1

        return bust_types

    def run(self):
        """
        ######################################
        #                                    #
        #            MAIN ROUTINE            #
        #                                    #
        ######################################
        Principle code that drives TAF Monitor once a bench has been selected.
        """
        # airfields = airfield_benches[self.bench_selected]

        # Lists to record details of broken tafs.
        bust_summaries = {'wind': [], 'visibility': [], 'cloud': [],
                          'weather': [], 'all': [], 'metars_used': []}

        # To store latest bust times (used to limit number of busts per hour)
        latest_busts = {'wind': False, 'visibility': False, 'cloud': False,
                        'weather': False}

        # Day TAF starts
        taf_start_day = self.taf_start_time.day

        # Check each METAR for busts
        for ind, metar in enumerate(self.metars):

            # Check if airfield is helicopter station.
            airfield = all_airfields[self.icao]

            # Simplify TAF groups to just BECMG and TEMPO.
            s_taf = Simplify_TAF().by_group(self.taf.copy())

            # Day METAR valid
            try:
                metar_day = int(metar[1][:-1][:2])
            except:
                print('Problem with METAR day', metar)
                continue

            # Determine METAR month and year from TAF
            if metar_day == self.taf_start_time.day:
                metar_month = self.taf_start_time.month
                metar_year = self.taf_start_time.year
            else:
                metar_month = (self.taf_start_time + timedelta(days=1)).month
                metar_year = (self.taf_start_time + timedelta(days=1)).year

            # Get METAR issue time as python datetime object.
            try:
                metar_time = ConstructTimeObject(
                    metar[1][:-1], metar_day, metar_month, metar_year
                    ).METAR()
            # Move to next METAR if any errors encountered
            except:
                print('Problem with METAR time: ', metar)
                continue

            # Stops checking TAF if it is not currently valid.
            if not Is_Time_Current(self.taf_start_time, self.taf_end_time,
                                   metar_time).check():
                continue

            # Get indices of different groups.
            [
                becmg_groups,
                tempo_groups,
                change_groups,
            ] = Interpret_TAF().get_all_indices(s_taf)

            # Create dict types to contain conditions.
            self.metar_conditions = DefineConditionsDictionary().create(
                condition_type
            )
            self.base_conditions = DefineConditionsDictionary().create(
                condition_type
            )
            self.base_conditions_2 = DefineConditionsDictionary().create(
                condition_type
            )
            self.becmg_changes = DefineConditionsDictionary().create(condition_type)
            self.becmg_period = DefineConditionsDictionary().create(condition_type)
            self.tempo_changes = DefineConditionsDictionary().create(condition_type)
            self.taf_status = DefineConditionsDictionary().create(condition_type)

            # Get base conditions of taf.
            self.base_conditions = Interpret_TAF().conditions(
                s_taf[0 : change_groups[0]], self.base_conditions, self.icao
            )

            # Check becmg group status and modify taf.
            [
                s_taf,
                kill_list,
                self.becmg_changes,
                self.becmg_period
            ] = Interpret_TAF().interpret_becmg_group(
                s_taf,
                becmg_groups,
                metar_time,
                change_groups,
                self.becmg_changes,
                self.becmg_period,
                self.icao,
                self.taf_start_time.day,
                self.taf_start_time.month,
                self.taf_start_time.year,
            )

            # Remove becmg groups that have not yet started.
            s_taf = Simplify_TAF().kill_elements(s_taf, kill_list)

            # Apply changes from completed becmg groups to the base
            # conditions.
            self.base_conditions = Simplify_TAF().apply_becmg(
                self.base_conditions, self.becmg_changes, condition_type
            )
            self.base_conditions_2 = Simplify_TAF().apply_becmg(
                self.base_conditions_2, self.becmg_period, condition_type
            )

            # Remake indices for remaining tempo groups (all groups are now
            # of type tempo).
            [
                becmg_groups,
                tempo_groups,
                change_groups,
            ] = Interpret_TAF().get_all_indices(s_taf)

            # Get tempo groups valid at current metar time.
            self.tempo_changes = Simplify_TAF().get_valid_tempo_groups(
                s_taf,
                tempo_groups,
                metar_time,
                change_groups,
                self.tempo_changes,
                condition_type,
                self.icao,
                self.taf_start_time.day,
                self.taf_start_time.month,
                self.taf_start_time.year,
            )

            # Extract current METAR conditions.
            self.metar_conditions = Interpret_TAF().conditions(
                metar, self.metar_conditions, self.icao
            )

            # Check METAR has three essential components; wind, vis, cloud.
            # CAVOK dealt with in get_conditions.
            valid_metar = True
            for item_mt in ["wind", "visibility", "cloud"]:
                if not self.metar_conditions[item_mt]:
                    valid_metar = False
                    break

            # Move to next METAR if not valid
            if not valid_metar:
                continue

            # Calls a checking routine for each of the condition_type to
            # determine taf status.
            wind_bust_types = self.check_condition(wind, "wind", airfield)
            vis_bust_types = self.check_condition(visibility, "visibility",
                                                 airfield)
            wx_bust_types = self.check_condition(weather, "weather", airfield)
            cld_bust_types = self.check_condition(cloud, "cloud", airfield)

            # Collect bust_types into dictionary
            all_bust_types = {'wind': wind_bust_types,
                              'visibility': vis_bust_types,
                              'cloud': cld_bust_types,
                              'weather': wx_bust_types}

            # To keep track of multiple busts per METAR
            bust_list, metar_used = [], True

            for item in condition_type:

                # If TAF is not bust, move to next iteration
                if self.taf_status[item]:
                    continue

                # If more than 20 minutes after latest bust, append to
                # appropriate list, otherwise move to next item
                if not latest_busts[item]:
                    pass
                elif (metar_time -
                      latest_busts[item]).total_seconds() / 60 > 20:
                    pass
                else:
                    metar_used = False
                    continue

                # Append to bust lists
                bust_summaries[item].append([all_bust_types[item], metar, 
                                             metar_time])
                bust_list.append(item)

                # Update latest bust time
                latest_busts[item] = metar_time

            # Add to all bust list if necessary
            if bust_list:
                bust_summaries['all'].append([bust_list, metar])

            # Add to metars used count
            if metar_used:
                bust_summaries['metars_used'].append(metar)

        return bust_summaries
