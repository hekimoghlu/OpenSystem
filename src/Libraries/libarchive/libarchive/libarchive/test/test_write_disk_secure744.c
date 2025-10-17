/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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

#define UMASK 022

/*
 * Github Issue #744 describes a bug in the sandboxing code that
 * causes very long pathnames to not get checked for symlinks.
 */

DEFINE_TEST(test_write_disk_secure744)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	skipping("archive_write_disk security checks not supported on Windows");
#else
	struct archive *a;
	struct archive_entry *ae;
	size_t buff_size = 8192;
	char *buff = malloc(buff_size);
	char *p = buff;
	int n = 0;
	int t;

	assert(buff != NULL);

	/* Start with a known umask. */
	assertUmask(UMASK);

	/* Create an archive_write_disk object. */
	assert((a = archive_write_disk_new()) != NULL);
	archive_write_disk_set_options(a, ARCHIVE_EXTRACT_SECURE_SYMLINKS);

	while (p + 500 < buff + buff_size) {
		memset(p, 'x', 100);
		p += 100;
		p[0] = '\0';

		buff[0] = ((n / 1000) % 10) + '0';
		buff[1] = ((n / 100) % 10)+ '0';
		buff[2] = ((n / 10) % 10)+ '0';
		buff[3] = ((n / 1) % 10)+ '0';
		buff[4] = '_';
		++n;

		/* Create a symlink pointing to the testworkdir */
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, buff);
		archive_entry_set_mode(ae, S_IFREG | 0777);
		archive_entry_copy_symlink(ae, testworkdir);
		assertEqualIntA(a, ARCHIVE_OK, archive_write_header(a, ae));
		archive_entry_free(ae);

		*p++ = '/';
		snprintf(p, buff_size - (p - buff), "target%d", n);

		/* Try to create a file through the symlink, should fail. */
		assert((ae = archive_entry_new()) != NULL);
		archive_entry_copy_pathname(ae, buff);
		archive_entry_set_mode(ae, S_IFDIR | 0777);

		t = archive_write_header(a, ae);
		archive_entry_free(ae);
		failure("Attempt to create target%d via %d-character symlink should have failed", n, (int)strlen(buff));
		if(!assertEqualInt(ARCHIVE_FAILED, t)) {
			break;
		}
	}
	archive_free(a);
	free(buff);
#endif
}
