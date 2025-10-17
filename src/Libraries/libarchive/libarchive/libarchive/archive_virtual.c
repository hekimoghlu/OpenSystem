/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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

#include "archive.h"
#include "archive_entry.h"
#include "archive_private.h"
#ifdef __APPLE__
#include "archive_check_entitlement.h"
#endif

int
archive_filter_code(struct archive *a, int n)
{
	return ((a->vtable->archive_filter_code)(a, n));
}

int
archive_filter_count(struct archive *a)
{
	return ((a->vtable->archive_filter_count)(a));
}

const char *
archive_filter_name(struct archive *a, int n)
{
	return ((a->vtable->archive_filter_name)(a, n));
}

la_int64_t
archive_filter_bytes(struct archive *a, int n)
{
	return ((a->vtable->archive_filter_bytes)(a, n));
}

int
archive_free(struct archive *a)
{
	if (a == NULL)
		return (ARCHIVE_OK);
#ifdef __APPLE__
	archive_entitlement_cleanup();
#endif
	return ((a->vtable->archive_free)(a));
}

int
archive_write_close(struct archive *a)
{
	return ((a->vtable->archive_close)(a));
}

int
archive_read_close(struct archive *a)
{
	return ((a->vtable->archive_close)(a));
}

int
archive_write_fail(struct archive *a)
{
	a->state = ARCHIVE_STATE_FATAL;
	return a->state;
}

int
archive_write_free(struct archive *a)
{
	return archive_free(a);
}

#if ARCHIVE_VERSION_NUMBER < 4000000
/* For backwards compatibility; will be removed with libarchive 4.0. */
int
archive_write_finish(struct archive *a)
{
	return archive_write_free(a);
}
#endif

int
archive_read_free(struct archive *a)
{
	return archive_free(a);
}

#if ARCHIVE_VERSION_NUMBER < 4000000
/* For backwards compatibility; will be removed with libarchive 4.0. */
int
archive_read_finish(struct archive *a)
{
	return archive_read_free(a);
}
#endif

int
archive_write_header(struct archive *a, struct archive_entry *entry)
{
	++a->file_count;
	return ((a->vtable->archive_write_header)(a, entry));
}

int
archive_write_finish_entry(struct archive *a)
{
	return ((a->vtable->archive_write_finish_entry)(a));
}

la_ssize_t
archive_write_data(struct archive *a, const void *buff, size_t s)
{
	return ((a->vtable->archive_write_data)(a, buff, s));
}

la_ssize_t
archive_write_data_block(struct archive *a, const void *buff, size_t s,
    la_int64_t o)
{
	if (a->vtable->archive_write_data_block == NULL) {
		archive_set_error(a, ARCHIVE_ERRNO_MISC,
		    "archive_write_data_block not supported");
		a->state = ARCHIVE_STATE_FATAL;
		return (ARCHIVE_FATAL);
	}
	return ((a->vtable->archive_write_data_block)(a, buff, s, o));
}

int
archive_read_next_header(struct archive *a, struct archive_entry **entry)
{
	return ((a->vtable->archive_read_next_header)(a, entry));
}

int
archive_read_next_header2(struct archive *a, struct archive_entry *entry)
{
	return ((a->vtable->archive_read_next_header2)(a, entry));
}

int
archive_read_data_block(struct archive *a,
    const void **buff, size_t *s, la_int64_t *o)
{
	return ((a->vtable->archive_read_data_block)(a, buff, s, o));
}
