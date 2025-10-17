/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include "archive_platform.h"

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#include "archive_write_private.h"

static int
set_passphrase(struct archive_write *a, const char *p)
{
	if (p == NULL || p[0] == '\0') {
		archive_set_error(&a->archive, ARCHIVE_ERRNO_MISC,
		    "Empty passphrase is unacceptable");
		return (ARCHIVE_FAILED);
	}
	free(a->passphrase);
	a->passphrase = strdup(p);
	if (a->passphrase == NULL) {
		archive_set_error(&a->archive, ENOMEM,
		    "Can't allocate data for passphrase");
		return (ARCHIVE_FATAL);
	}
	return (ARCHIVE_OK);
}


int
archive_write_set_passphrase(struct archive *_a, const char *p)
{
	struct archive_write *a = (struct archive_write *)_a;

	archive_check_magic(_a, ARCHIVE_WRITE_MAGIC, ARCHIVE_STATE_NEW,
		"archive_write_set_passphrase");

	return (set_passphrase(a, p));
}


int
archive_write_set_passphrase_callback(struct archive *_a, void *client_data,
    archive_passphrase_callback *cb)
{
	struct archive_write *a = (struct archive_write *)_a;

	archive_check_magic(_a, ARCHIVE_WRITE_MAGIC, ARCHIVE_STATE_NEW,
		"archive_write_set_passphrase_callback");

	a->passphrase_callback = cb;
	a->passphrase_client_data = client_data;
	return (ARCHIVE_OK);
}


const char *
__archive_write_get_passphrase(struct archive_write *a)
{

	if (a->passphrase != NULL)
		return (a->passphrase);

	if (a->passphrase_callback != NULL) {
		const char *p;
		p = a->passphrase_callback(&a->archive,
		    a->passphrase_client_data);
		set_passphrase(a, p);
		a->passphrase_callback = NULL;
		a->passphrase_client_data = NULL;
	}
	return (a->passphrase);
}
