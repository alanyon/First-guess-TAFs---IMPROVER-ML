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

import os
import re

try:
    from urllib.request import urlopen as urlopen
except ImportError:
    from urllib2 import urlopen as urlopen

# from urllib.request import urlopen
import taf_monitor.weather_regex as wr

# Proxy servers are external to Met Office, so we disable these to be able
# to request internal pages.
del os.environ['http_proxy']


class RetrieveObservations(object):
    """
    Observation retrieval from MetDB using a web interface as there is no
    access to the metdb python modules in many FSD sites, including Exeter
    ops.

    INPUTS
    ------
    ICAO - airfield identifier
    TORM - type of data requested (TAF, METAR, or SPECI).

    RETURNS
    -------
    List of elements comprising requested data.
    """

    def __init__(
        self, ICAO, TORM, time_now, latest_only=True, start_time=None, end_time=None
    ):
        self.ICAO = ICAO
        self.TORM = TORM
        self.latest_only = latest_only
        if not latest_only:
            if start_time is None or end_time is None:
                raise ValueError(
                    "For range retrieval a start_time and " "end_time must be specified"
                )
            self.specify_latest = ""
            self.start_date = start_time.strftime("%Y%m%d")
            self.start_hour = start_time.hour
            self.end_date = end_time.strftime("%Y%m%d")
            self.end_hour = end_time.hour

        else:
            self.specify_latest = "&submit=Retrieve+Latest+Report"
            self.start_date = time_now.strftime("%Y%m%d")
            self.start_hour = 0
            self.end_date = time_now.strftime("%Y%m%d")
            self.end_hour = 0
        self.metdb = (
            "http://mdbdb-prod/cgi-bin/moods/webret.pl?pageType="
            "mainpage&subtype={type}&system=mdbdb-prod&idType"
            "=ICAO&stn01={icao}&stn02=&stn03=&stn04=&stn05=&stn06"
            "=&stn07=&stn08=&stn09=&stn10={latest}&startdate={start_date}"
            "&starthour={start_hour:02d}&startminute=00"
            "&enddate={end_date}&endhour={end_hour:02d}&endminute=59&maxobs="
            "0500&header=no&area_n=&area_s=&area_w=&area_e="
        )

    def request_type(self):
        """
        Uses input flag to determine which type of record is being requested.

        """
        if self.TORM == "T":
            return "TAFS"
        elif self.TORM == "M":
            return "METARS"
        elif self.TORM == "S":
            return "SPECI"
        else:
            msg = "Unknown record requested: Not TAF/METAR/SPECI"
            raise ValueError(msg)

    def retrieve_record(self, MTtarget):
        URLsub = self.metdb.format(
            type=MTtarget,
            icao=self.ICAO,
            latest=self.specify_latest,
            start_date=self.start_date,
            start_hour=self.start_hour,
            end_date=self.end_date,
            end_hour=self.end_hour,
        )
        raw_url_text = urlopen(URLsub).read().decode("utf-8")

        if not wr.VALID_DATETIME.search(raw_url_text):
            return ["NoRecord"]

        text = re.search(r"report\n.*?pre", raw_url_text, re.DOTALL).group()
        raw_url_text = text.replace("\n               ", "")
        returnterm = re.compile("[0-9]{6}" + "Z " + self.ICAO.upper() + ".*")
        returntext = re.findall(returnterm, raw_url_text)
        # Return tafs and metars as python lists.
        if self.latest_only:
            return returntext[-1].split()[2:]
        else:
            return [item.split()[2:] for item in returntext]

    # Retrieve latest METAR and TAF from MOODS, strip html and break down into
    # component parts.
    def operation(self):
        MTtarget = self.request_type()
        return self.retrieve_record(MTtarget)
