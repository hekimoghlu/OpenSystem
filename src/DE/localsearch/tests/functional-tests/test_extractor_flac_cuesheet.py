# Copyright (C) 2019, Sam Thursfield (sam@afuera.me.uk)
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.

"""
Tests the FLAC+cuesheet extraction feature.
"""


import pathlib
import shutil
import tempfile
import unittest as ut

import configuration as cfg
import datagenerator
import fixtures


class FlacCuesheetTest(fixtures.TrackerExtractTestCase):
    def spec(self, audio_path):
        audio_uri = audio_path.as_uri()
        return {
            "@type": ["nmm:MusicPiece", "nfo:Audio"],
            "nmm:trackNumber": 2,
            "nfo:audioOffset": 257.69333333333333,
            "nmm:performer": {"@id": "urn:artist:My%20Bloody%20Valentine"},
            "nie:isStoredAs": {
                "@id": audio_uri,
                "@type": ["nfo:Audio", "nmm:MusicPiece", "nfo:Audio"],
                "nmm:trackNumber": 1,
                "nfo:audioOffset": 0.0,
                "nmm:performer": {
                    "@id": "urn:artist:My%20Bloody%20Valentine",
                    "nmm:artistName": "My Bloody Valentine",
                    "@type": "nmm:Artist",
                },
                "nfo:channels": 1,
                "nie:isStoredAs": {"@id": audio_uri},
                "nfo:sampleRate": 44100,
                "nie:title": "Only Shallow",
                "nmm:musicAlbum": {
                    "@id": "urn:album:Loveless:My%20Bloody%20Valentine:1991-01-01"
                },
                "nmm:musicAlbumDisc": {
                    "@id": "urn:album-disc:Loveless:My%20Bloody%20Valentine:1991-01-01:Disc1"
                },
                "nfo:duration": 257,
            },
            "nie:title": "Loomer",
            "nfo:duration": 102,
            "nmm:musicAlbum": {
                "@id": "urn:album:Loveless:My%20Bloody%20Valentine:1991-01-01",
                "nie:title": "Loveless",
                "nmm:albumTrackCount": 2,
                "@type": "nmm:MusicAlbum",
                "nmm:albumArtist": {"@id": "urn:artist:My%20Bloody%20Valentine"},
            },
            "nmm:musicAlbumDisc": {
                "@id": "urn:album-disc:Loveless:My%20Bloody%20Valentine:1991-01-01:Disc1",
                "nmm:setNumber": 1,
                "nmm:albumDiscAlbum": {
                    "@id": "urn:album:Loveless:My%20Bloody%20Valentine:1991-01-01"
                },
                "@type": "nmm:MusicAlbumDisc",
            },
        }

    def test_external_cue_sheet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            datadir = pathlib.Path(__file__).parent.joinpath(
                "data", "extractor-content"
            )
            shutil.copy(datadir.joinpath("audio", "cuesheet-test.cue"), tmpdir)

            audio_path = pathlib.Path(tmpdir).joinpath("cuesheet-test.flac")
            datagenerator.create_test_flac(audio_path, duration=6 * 60)

            result = fixtures.get_tracker_extract_output(
                cfg.test_environment(tmpdir), audio_path, output_format="json-ld"
            )
            file_data = result["@graph"][0]

        self.assert_extract_result_matches_spec(
            self.spec(audio_path), file_data, audio_path, __file__
        )


if __name__ == "__main__":
    fixtures.tracker_test_main()
