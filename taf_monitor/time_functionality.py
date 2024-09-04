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

from datetime import datetime as dt
from datetime import time as timeobject
from datetime import timedelta as timedelta


class ConstructTimeObject(object):
    """
    Converts taf "1812/1815" style time groups into python datetime objects.

    """

    def __init__(self, time_in, current_day, current_month, current_year):
        self.time_in = time_in
        self.current_day = current_day
        self.current_month = current_month
        self.current_year = current_year

    def TAF(self):
        start_day = int(self.time_in[0:2])
        end_day = int(self.time_in[5:7])
        start_hour = int(self.time_in[2:4])
        end_hour = int(self.time_in[7:9])
        start_min = 0
        if end_hour == 24:
            end_hour = 23
            end_min = 59
        else:
            end_min = 0

        last_day_of_month = self.is_today_last_day_of_month()
        start_month = self.current_month
        end_month = self.current_month
        start_year = self.current_year
        end_year = self.current_year

        if last_day_of_month:
            next_month = (
                dt(self.current_year, self.current_month, self.current_day)
                + timedelta(days=1)
            ).month
            next_year = (
                dt(self.current_year, self.current_month, self.current_day)
                + timedelta(days=1)
            ).year

            # Checking on last day of month with TAF that bridges into the next
            if start_day == self.current_day and end_day == 1:
                end_month = next_month
                end_year = next_year

            # Checking at ~2355Z on last day of month with TAF that starts at
            # 00Z on the first day of the next.
            elif start_day == 1:
                start_month = next_month
                end_month = next_month
                start_year = next_year
                end_year = next_year

        elif self.current_day == 1:
            # Checking on the 1st of a new month with a TAF that started the
            # previous day.
            if start_day > 20:
                last_month = (
                    dt(self.current_year, self.current_month, self.current_day)
                    - timedelta(days=1)
                ).month
                last_year = (
                    dt(self.current_year, self.current_month, self.current_day)
                    - timedelta(days=1)
                ).year
                start_month = last_month
                # Year should only change if we are entering January.
                start_year = last_year
                if end_day == start_day:
                    end_month = last_month

        start_time = dt(start_year, start_month, start_day, start_hour, start_min)
        end_time = dt(end_year, end_month, end_day, end_hour, end_min)

        return start_time, end_time

    # Converts metar observation time into python datetime object. Adds 10
    # minutes so that a 0850Z metar falls within a DD09/DD10 taf element.
    def METAR(self):
        """
        Allow for case: 2350Z ob, checked at 0005Z with a month transition
        taking place at midnight. Without this alternative we would get back
        a day at the end of the new month, and possibly an error if the new
        month has fewer days than the one just finished, creating an invalid
        datetime object.

        """
        if self.current_day == 1 and not int(self.time_in[0:2]) == self.current_day:
            return (
                dt.combine(
                    dt(self.current_year, self.current_month, self.current_day)
                    - timedelta(days=1),
                    timeobject(int(self.time_in[2:4]), int(self.time_in[4:6])),
                )
            )
        else:
            return (
                dt(
                    self.current_year,
                    self.current_month,
                    int(self.time_in[0:2]),
                    int(self.time_in[2:4]),
                    int(self.time_in[4:6]),
                )
            )

    def is_today_last_day_of_month(self):
        if (
            not (
                dt(self.current_year, self.current_month, self.current_day)
                + timedelta(days=1)
            ).month
            == self.current_month
        ):
            return True
        else:
            return False


class Is_Time_Current(object):
    """
    Checks is a given time falls within two bracketing times.

    """

    def __init__(self, start_time, end_time, test_time):
        self.start_time = start_time
        self.end_time = end_time
        self.test_time = test_time

        if any(
            [
                check is not dt
                for check in [type(start_time), type(end_time), type(test_time)]
            ]
        ):
            msg = "Received invalid input: Not a datetime object."
            raise ValueError(msg)

    def check(self):
        if self.start_time <= self.test_time <= self.end_time:
            return True
        else:
            return False
