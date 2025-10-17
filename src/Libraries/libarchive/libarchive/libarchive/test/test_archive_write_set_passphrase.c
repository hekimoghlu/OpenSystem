/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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

struct archive_write;
extern const char * __archive_write_get_passphrase(struct archive_write *);

static void
test(int pristine)
{
	struct archive* a = archive_write_new();
	struct archive_write* aw = (struct archive_write *)a;

	if (!pristine) {
		archive_write_add_filter_gzip(a);
		archive_write_set_format_iso9660(a);
        }

	assertEqualInt(ARCHIVE_OK, archive_write_set_passphrase(a, "pass1"));
	/* An empty passphrase cannot be accepted. */
	assertEqualInt(ARCHIVE_FAILED, archive_write_set_passphrase(a, ""));
	/* NULL passphrases cannot be accepted. */
	assertEqualInt(ARCHIVE_FAILED, archive_write_set_passphrase(a, NULL));
	/* Check a passphrase. */
	assertEqualString("pass1", __archive_write_get_passphrase(aw));
	/* Change the passphrase. */
	assertEqualInt(ARCHIVE_OK, archive_write_set_passphrase(a, "pass2"));
	assertEqualString("pass2", __archive_write_get_passphrase(aw));

	archive_write_free(a);
}

DEFINE_TEST(test_archive_write_set_passphrase)
{
	test(1);
	test(0);
}


static const char *
callback1(struct archive *a, void *_client_data)
{
	int *cnt;

	(void)a; /* UNUSED */

	cnt = (int *)_client_data;
	*cnt += 1;
	return ("passCallBack");
}

DEFINE_TEST(test_archive_write_set_passphrase_callback)
{
	struct archive* a = archive_write_new();
	struct archive_write* aw = (struct archive_write *)a;
	int cnt = 0;

	archive_write_set_format_zip(a);

	assertEqualInt(ARCHIVE_OK,
	    archive_write_set_passphrase_callback(a, &cnt, callback1));
	/* Check a passphrase. */
	assertEqualString("passCallBack", __archive_write_get_passphrase(aw));
	assertEqualInt(1, cnt);
	/* Callback function should be called just once. */
	assertEqualString("passCallBack", __archive_write_get_passphrase(aw));
	assertEqualInt(1, cnt);

	archive_write_free(a);
}
