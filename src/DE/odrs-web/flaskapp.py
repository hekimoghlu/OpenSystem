#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2017 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+

from odrs import app

if __name__ == "__main__":
    app.debug = True
    app.run()
