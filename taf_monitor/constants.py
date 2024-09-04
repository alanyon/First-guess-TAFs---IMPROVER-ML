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

# Define types of conditions to be found.
condition_type = ["wind", "visibility", "weather", "cloud"]

# Define visibility types of weather.
# There are too many permutations of precipitation codes to explicitly list
# them. As such their presence in metars/tafs must be determined by the
# exclusion of vis types and non-standard groups (e.g. NSW).
vis_types = ["FG", "BR", "HZ", "FU", "DU", "PO", "SA", "VCFG", "BCFG", "MIFG", "PRFG"]
# Weather types that must be covered explicitly, otherwise the taf is bust.
special_types = [
    "TS",
    "SQ",
    "FC",
    "IC",
    "FZDZ",
    "FZRA",
    "FZFG",
    "+",
    "DRSN",
    "BLSN",
    "DRSN",
    "BLSN",
    "DRDU",
    "BLDU",
    "SS",
    "DS",
]

# Define times at which to run taf Monitor (ideally visual weather would push
#  new metars to the code).
early_time = 26
late_time = 56
