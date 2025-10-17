/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "test.h"
__FBSDID("$FreeBSD$");

void extract_bomb(const char *reffile)
{
    extract_reference_file(reffile);
    int r = systemf("%s xf %s", testprog, reffile);
    assertEqualInt(1, (r == 0) ? 0:1);
}

/* Uses overlapping files with mismatched filenames */
DEFINE_TEST(test_zip_bomb_overlap)
{
    extract_bomb("zbo.zip");
}

DEFINE_TEST(test_zip_bomb_small)
{
    extract_bomb("zbsm.zip");
}

/* Uses the "extra field" to improve compression ratio */
DEFINE_TEST(test_zip_bomb_small_extra)
{
    extract_bomb("zbsm.extra.zip");
}

DEFINE_TEST(test_zip_bomb_large)
{
    extract_bomb("zblg.zip");
}

/* Uses the "extra field" to improve compression ratio */
DEFINE_TEST(test_zip_bomb_large_extra)
{
    extract_bomb("zblg.extra.zip");
}

DEFINE_TEST(test_zip_bomb_extralarge)
{
    extract_bomb("zbxl.zip");
}

/* Uses the "extra field" to improve compression ratio */
DEFINE_TEST(test_zip_bomb_extralarge_extra)
{
    extract_bomb("zbxl.extra.zip");
}

/* Uses bzip2 algorithm instead of DEFLATE */
DEFINE_TEST(test_zip_bomb_bzip2)
{
    extract_bomb("zbbz2.zip");
}
