/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

DEFINE_TEST(test_archive_api_feature)
{
	char buff[128];
	const char *p;

	/* This is the (hopefully) final versioning API. */
	assertEqualInt(ARCHIVE_VERSION_NUMBER, archive_version_number());
	snprintf(buff, sizeof(buff), "libarchive %d.%d.%d",
	    archive_version_number() / 1000000,
	    (archive_version_number() / 1000) % 1000,
	    archive_version_number() % 1000);
	failure("Version string is: %s, computed is: %s",
	    archive_version_string(), buff);
	assertEqualMem(buff, archive_version_string(), strlen(buff));
	if (strlen(buff) < strlen(archive_version_string())) {
		p = archive_version_string() + strlen(buff);
		failure("Version string is: %s", archive_version_string());
		if (p[0] == 'd'&& p[1] == 'e' && p[2] == 'v')
			p += 3;
		else {
			assert(*p == 'a' || *p == 'b' || *p == 'c' || *p == 'd');
			++p;
		}
		failure("Version string is: %s", archive_version_string());
		assert(*p == '\0');
	}
}
