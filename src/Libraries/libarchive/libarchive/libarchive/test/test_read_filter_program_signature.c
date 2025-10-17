/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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

static unsigned char archive[] = {
31,139,8,0,222,'C','p','C',0,3,211,'c',160,'=','0','0','0','0','7','5','U',
0,210,134,230,166,6,200,'4',28,'(',24,26,24,27,155,24,152,24,154,27,155,')',
24,24,26,152,154,25,'2','(',152,210,193,'m',12,165,197,'%',137,'E','@',167,
148,'d',230,226,'U','G','H',30,234,15,'8','=',10,'F',193,'(',24,5,131,28,
0,0,29,172,5,240,0,6,0,0};
#define signature archive
#define badsignature (&archive[1])
#define signature_len 4

DEFINE_TEST(test_read_filter_program_signature)
{
	int r;
	struct archive_entry *ae;
	struct archive *a;

	/*
	 * First, test handling when a non-existent compression
	 * program is requested.
	 */
	assert((a = archive_read_new()) != NULL);
	r = archive_read_support_filter_program(a, "nonexistent");
	if (r == ARCHIVE_FATAL) {
		skipping("archive_read_support_filter_program() "
		    "unsupported on this platform");
		return;
	}
	assertEqualIntA(a, ARCHIVE_OK, r);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_FATAL,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * If we have "gzip -d", try using that.
	 */
	if (!canGzip()) {
		skipping("Can't run gzip program on this platform");
		return;
	}
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_filter_program_signature(a, "gzip -d",
		signature, signature_len));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_next_header(a, &ae));
	assertEqualInt(archive_filter_code(a, 0), ARCHIVE_FILTER_PROGRAM);
	assertEqualInt(archive_format(a), ARCHIVE_FORMAT_TAR_USTAR);
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));

	/*
	 * Test bad signature.
	 */
	assert((a = archive_read_new()) != NULL);
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_filter_none(a));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_filter_program_signature(a, "gunzip",
		badsignature, signature_len));
	assertEqualIntA(a, ARCHIVE_OK,
	    archive_read_support_format_all(a));
	assertEqualIntA(a, ARCHIVE_FATAL,
	    archive_read_open_memory(a, archive, sizeof(archive)));
	assertEqualIntA(a, ARCHIVE_OK, archive_read_close(a));
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
