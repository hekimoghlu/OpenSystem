#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Richard Hughes <richard@hughsie.com>
#
# SPDX-License-Identifier: GPL-3.0+
#
# pylint: disable=no-self-use,protected-access,wrong-import-position

import os
import sys

import unittest

# allows us to run this from the project root
sys.path.append(os.path.realpath("."))

from odrs.util import json_success, json_error, _locale_is_compatible
from odrs.util import _get_user_key, _password_hash
from odrs.util import _sanitised_version, _sanitised_summary, _sanitised_description


class UtilTest(unittest.TestCase):
    def test_sanitise(self):

        self.assertEqual(_sanitised_version("16.12.3"), "16.12.3")
        self.assertEqual(_sanitised_version("0:1.2.3+rh"), "1.2.3")
        self.assertEqual(_sanitised_version("16.11.0~ds0"), "16.11.0")
        self.assertEqual(
            _sanitised_summary("   not sure why people include.   "),
            "not sure why people include",
        )
        self.assertEqual(
            _sanitised_description("   this is awesome :) !!   "), "this is awesome !!"
        )

    def test_response(self):

        self.assertEqual(str(json_success("ok")), "<Response 40 bytes [200 OK]>")
        self.assertEqual(
            str(json_error("nok")), "<Response 42 bytes [400 BAD REQUEST]>"
        )

    def test_locale(self):

        self.assertTrue(_locale_is_compatible("en_GB", "en_GB"))
        self.assertTrue(_locale_is_compatible("en_GB", "en_AU"))
        self.assertTrue(_locale_is_compatible("en_GB", "C"))
        self.assertTrue(_locale_is_compatible("C", "en_GB"))
        self.assertFalse(_locale_is_compatible("fr_FR", "en_GB"))

    def test_user_key(self):

        os.environ["ODRS_REVIEWS_SECRET"] = "1"
        self.assertEqual(
            _get_user_key("foo", "gimp.desktop"),
            "8d68a9e8054a18cb11e62242f9036aca786551d8",
        )

    def test_legacy_hash(self):

        self.assertEqual(
            _password_hash("foo"), "9cab340b3184a1f792d6629806703aed450ecd48"
        )


if __name__ == "__main__":
    unittest.main()
