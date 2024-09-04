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


def define_thresholds(rules="civil", helicopter=False):
    """
    Define the thresholds for visibilities and clouds. The lists include the
    extremes and lower-bound of each group.

    Keyword Args:
        rules (str):
            Determines which set of thresholds should be returned. Options
            are::

              'civil' - civil aviation thresholds.
              'military' - military aviation thresholds.

        helicopter (bool):
            For civil aviation there are some airfields with different
            thresholds because they operate significant numbers of
            helicopter flights. If helicopter=True these thresholds are
            returned.
    Returns
        tuple:
            Returns a tuple containing the threshold lists defined here.
    """
    if rules == "military":
        vis_thresholds = [0, 800, 1600, 2500, 3700, 5000, 8000, 10000]
        cloud_thresholds = [0, 2, 3, 5, 7, 15, 25, 50]
    elif rules == "civil":
        if helicopter:
            vis_thresholds = [0, 350, 800, 1500, 3000, 5000, 7000, 9998, 10000]
            cloud_thresholds = [0, 2, 5, 7, 10, 15, 50]
        else:
            vis_thresholds = [0, 350, 800, 1500, 5000, 9998, 10000]
            cloud_thresholds = [0, 2, 5, 10, 15, 50]
    else:
        raise ValueError("Unknown aviation rules requested.")

    return vis_thresholds, cloud_thresholds


def define_benches():
    """
    Assign a list of airfield ICAOs to a set of named benches. The required
    bench is chosen by the user, allowing them to only monitor a subset of
    airfields.

    Returns
        airfield_benches (dict):
            Each dictionary key corresponds to a named bench. The values for
            each bench are a list of ICAO that define the airfields that the
            bench is responsible for.
    """
    airfield_benches = {}
    # Define list of ICAO for each bench

    airfield_benches["Aviation Team Leader"] = ["EGGD", "EGGW", "EGSS", "EGFF"]

    airfield_benches["Aviation 1"] = [
        "EGBB",
        "EGNX",
        "EGHC",
        "EGTE",
        "EGHQ",
        "EGHE",
        "EGBJ",
        "EGHH",
        "EGHI",
        "EGSY",
    ]

    airfield_benches["Aviation 2"] = [
        "EGNH",
        "EGCK",
        "EGNC",
        "EGCN",
        "EGNV",
        "EGNR",
        "EGNJ",
        "EGGP",
        "EGCC",
        "EGNT",
        "EGNO",
        "EGSH",
        "EGNM",
    ]

    airfield_benches["Aviation 3"] = [
        "EGPD",
        "EGPI",
        "EGPL",
        "EGEC",
        "EGPN",
        "EGPH",
        "EGPF",
        "EGPE",
        "EGPA",
        "EGEO",
        "EGPK",
        "EGPM",
        "EGPO",
        "EGPB",
        "EGPU",
        "EGPC",
    ]

    airfield_benches["Heathrow Only"] = ["EGLL"]

    airfield_benches["Heathrow OpMet"] = [
        "EGKB",
        "EGLF",
        "EGKK",
        "EGLC",
        "EGMD",
        "EGKA",
        "EGMC",
        "EGTC",
        "EGTK",
        "EGSC",
        "EGLL",
    ]

    airfield_benches["St Helena"] = [
        "FHSH",
    ]

    airfield_benches["Northern Ireland"] = ["EGAA", "EGAE", "EGAC"]
    airfield_benches["Channel Islands"] = ["EGJJ", "EGJB"]

    # Military benches, DGU
    airfield_benches["DGU"] = ["EGWC", "EGOW", "EGXT", "EGOM", "EGOT"]

    # Create area "All Civilian Airfields" to cover entire country.
    all_areas = [
        item
        for key, sublist in list(airfield_benches.items())
        for item in sublist
        if key != "DGU"
    ]
    all_areas = list(dict.fromkeys(all_areas))
    airfield_benches["All Civilian Airfields"] = all_areas

    return airfield_benches


def define_airfields():
    """
    A complete list of airfields that are served by the Op centre. For each
    airfield a dictionary entry records whether the airfield uses helicopter
    thresholds, and also whether it can be issued a taf following a single
    metar.

    Returns
        airfields (dict):
            A dictionary with an entry for each airfield. The sub-keys are
            'helicopter', true if the site uses helicopter thresholds, and
            'single_metar_issue', true if only one metar is required to issue
            the airfield a taf.
    """
    airfields = {}

    # Civil airfields

    airfields["EGFF"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGBB"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNX"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGHE"] = {
        "helicopter": False,
        "single_metar_issue": True,
        "rules": "civil",
    }
    airfields["EGTE"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGHH"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGBJ"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGHQ"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGTK"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGHI"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGHC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGSH"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGTC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGSC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGSY"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPI"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGEC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNT"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGCN"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNJ"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNH"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNV"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNO"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGCC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGGP"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNR"] = {
        "helicopter": False,
        "single_metar_issue": True,
        "rules": "civil",
    }
    airfields["EGCK"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPM"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPB"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPA"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPC"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPE"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPD"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPN"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPH"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPO"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPU"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPL"] = {
        "helicopter": True,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGEO"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPF"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGPK"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGMC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGKB"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGLC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGLF"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGKA"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGMD"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGSS"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGGW"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGNM"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGGD"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGSS"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGGW"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGLL"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGKK"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGLC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGAA"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGAC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGAE"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGJJ"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }
    airfields["EGJB"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }

    # Military airfields

    airfields["EGWC"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "military",
    }
    airfields["EGOW"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "military",
    }
    airfields["EGXT"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "military",
    }
    airfields["EGOM"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "military",
    }
    airfields["EGOT"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "military",
    }

    # St Helena
    airfields["FHSH"] = {
        "helicopter": False,
        "single_metar_issue": False,
        "rules": "civil",
    }

    return airfields
