/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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

DEFINE_TEST(test_warn_missing_hardlink_target)
{
	struct archive *a;
	struct archive_entry *ae;

	assert(NULL != (a = archive_write_disk_new()));
	assert(NULL != (ae = archive_entry_new()));

	archive_entry_set_pathname(ae, "hardlink-name");
	archive_entry_set_hardlink(ae, "hardlink-target");

	assertEqualInt(ARCHIVE_FAILED, archive_write_header(a, ae));
	assertEqualInt(ENOENT, archive_errno(a));
	assertEqualString("Hard-link target 'hardlink-target' does not exist.",
	    archive_error_string(a));

	archive_entry_free(ae);
	archive_free(a);
}
