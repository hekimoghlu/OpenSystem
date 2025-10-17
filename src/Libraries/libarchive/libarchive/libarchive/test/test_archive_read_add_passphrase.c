/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

struct archive_read;
extern void __archive_read_reset_passphrase(struct archive_read *);
extern const char * __archive_read_next_passphrase(struct archive_read *);

static void
test(int pristine)
{
	struct archive* a = archive_read_new();

	if (!pristine) {
		archive_read_support_filter_all(a);
		archive_read_support_format_all(a);
        }

	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass1"));
	/* An empty passphrase cannot be accepted. */
	assertEqualInt(ARCHIVE_FAILED, archive_read_add_passphrase(a, ""));
	/* NULL passphrases cannot be accepted. */
	assertEqualInt(ARCHIVE_FAILED, archive_read_add_passphrase(a, NULL));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_add_passphrase)
{
	test(1);
	test(0);
}

DEFINE_TEST(test_archive_read_add_passphrase_incorrect_sequance)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;

	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass1"));

	/* No call of __archive_read_reset_passphrase() leads to
	 * get NULL even if a user has passed a passphrases. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_add_passphrase_single)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;

	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass1"));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "pass1" as a passphrase. */
	assertEqualString("pass1", __archive_read_next_passphrase(ar));
	/* Second call, we should get NULL which means all the passphrases
	 * are passed already. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_add_passphrase_multiple)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;

	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass1"));
	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass2"));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "pass1" as a passphrase. */
	assertEqualString("pass1", __archive_read_next_passphrase(ar));
	/* Second call, we should get "pass2" as a passphrase. */
	assertEqualString("pass2", __archive_read_next_passphrase(ar));
	/* Third call, we should get NULL which means all the passphrases
	 * are passed already. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

static const char *
callback1(struct archive *a, void *_client_data)
{
	(void)a; /* UNUSED */
	(void)_client_data; /* UNUSED */
	return ("passCallBack");
}

DEFINE_TEST(test_archive_read_add_passphrase_set_callback1)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;

	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_passphrase_callback(a, NULL, callback1));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	/* Second call, we still get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));

	archive_read_free(a);

	/* Without __archive_read_reset_passphrase call, the callback
	 * should work fine. */
	a = archive_read_new();
	ar = (struct archive_read *)a;
	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_passphrase_callback(a, NULL, callback1));
	/* Fist call, we should get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	/* Second call, we still get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

static const char *
callback2(struct archive *a, void *_client_data)
{
	int *cd = (int *)_client_data;

	(void)a; /* UNUSED */

	if (*cd == 0) {
		*cd = 1;
		return ("passCallBack");
	}
	return (NULL);
}

DEFINE_TEST(test_archive_read_add_passphrase_set_callback2)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;
	int client_data = 0;

	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_passphrase_callback(a, &client_data, callback2));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	/* Second call, we should get NULL which means all the passphrases
	 * are passed already. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_add_passphrase_set_callback3)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;
	int client_data = 0;

	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_passphrase_callback(a, &client_data, callback2));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	__archive_read_reset_passphrase(ar);
	/* After reset passphrase, we should get "passCallBack" passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	/* Second call, we should get NULL which means all the passphrases
	 * are passed already. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_add_passphrase_multiple_with_callback)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;
	int client_data = 0;

	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass1"));
	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass2"));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_passphrase_callback(a, &client_data, callback2));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "pass1" as a passphrase. */
	assertEqualString("pass1", __archive_read_next_passphrase(ar));
	/* Second call, we should get "pass2" as a passphrase. */
	assertEqualString("pass2", __archive_read_next_passphrase(ar));
	/* Third call, we should get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	/* Fourth call, we should get NULL which means all the passphrases
	 * are passed already. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

DEFINE_TEST(test_archive_read_add_passphrase_multiple_with_callback2)
{
	struct archive* a = archive_read_new();
	struct archive_read *ar = (struct archive_read *)a;
	int client_data = 0;

	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass1"));
	assertEqualInt(ARCHIVE_OK, archive_read_add_passphrase(a, "pass2"));
	assertEqualInt(ARCHIVE_OK,
	    archive_read_set_passphrase_callback(a, &client_data, callback2));

	__archive_read_reset_passphrase(ar);
	/* Fist call, we should get "pass1" as a passphrase. */
	assertEqualString("pass1", __archive_read_next_passphrase(ar));
	/* Second call, we should get "pass2" as a passphrase. */
	assertEqualString("pass2", __archive_read_next_passphrase(ar));
	/* Third call, we should get "passCallBack" as a passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));

	__archive_read_reset_passphrase(ar);
	/* After reset passphrase, we should get "passCallBack" passphrase. */
	assertEqualString("passCallBack", __archive_read_next_passphrase(ar));
	/* Second call, we should get "pass1" as a passphrase. */
	assertEqualString("pass1", __archive_read_next_passphrase(ar));
	/* Third call, we should get "passCallBack" as a passphrase. */
	assertEqualString("pass2", __archive_read_next_passphrase(ar));
	/* Fourth call, we should get NULL which means all the passphrases
	 * are passed already. */
	assertEqualString(NULL, __archive_read_next_passphrase(ar));

	archive_read_free(a);
}

