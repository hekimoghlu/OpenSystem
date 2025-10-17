/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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

DEFINE_TEST(test_read_format_rar_filter)
{
    const char *refname = "test_read_format_rar_filter.rar";
    struct archive *a;
    struct archive_entry *ae;
    char *buff[12];
    const char signature[12] = {
        0x4d, 0x5a, 0x90, 0x00,
        0x03, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00,
    };

    extract_reference_file(refname);
    assert((a = archive_read_new()) != NULL);
    assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
    assertEqualIntA(a, ARCHIVE_OK, archive_read_support_filter_all(a));
    assertEqualIntA(a, ARCHIVE_OK, archive_read_open_filename(a, refname, 10240));

    assertA(0 == archive_read_next_header(a, &ae));
    assertEqualString("bsdcat.exe", archive_entry_pathname(ae));
    assertA((int)archive_entry_mtime(ae));
    assertEqualInt(204288, archive_entry_size(ae));
    assertA(12 == archive_read_data(a, buff, 12));
    assertEqualMem(buff, signature, 12);

    assertA(1 == archive_read_next_header(a, &ae));
    assertEqualInt(1, archive_file_count(a));
    assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
    assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
