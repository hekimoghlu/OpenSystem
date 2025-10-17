/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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

DEFINE_TEST(test_read_too_many_filters)
{
	const char *name = "test_read_too_many_filters.gz";
	struct archive *a;
	int r;

	assert((a = archive_read_new()) != NULL);
	r = archive_read_support_filter_gzip(a);
	if (r == ARCHIVE_WARN) {
		skipping("gzip reading not fully supported on this platform");
	}
	assertEqualIntA(a, ARCHIVE_OK, archive_read_support_format_all(a));
	extract_reference_file(name);
	assertEqualIntA(a, ARCHIVE_FATAL,
	    archive_read_open_filename(a, name, 200));

	// Can't assert the return value here:
	//  = Decompressing via zlib will return ARCHIVE_OK
	//  = Decompressing via external gzip will return ARCHIVE_WARN
	//    (Due to a dirty shutdown of the gzip program.)
	archive_read_close(a);
	assertEqualInt(ARCHIVE_OK, archive_read_free(a));
}
