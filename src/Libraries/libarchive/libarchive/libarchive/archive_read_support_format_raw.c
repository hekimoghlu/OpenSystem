/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "archive.h"
#include "archive_entry.h"
#include "archive_private.h"
#include "archive_read_private.h"

struct raw_info {
	int64_t offset; /* Current position in the file. */
	int64_t unconsumed;
	int     end_of_file;
};

static int	archive_read_format_raw_bid(struct archive_read *, int);
static int	archive_read_format_raw_cleanup(struct archive_read *);
static int	archive_read_format_raw_read_data(struct archive_read *,
		    const void **, size_t *, int64_t *);
static int	archive_read_format_raw_read_data_skip(struct archive_read *);
static int	archive_read_format_raw_read_header(struct archive_read *,
		    struct archive_entry *);

int
archive_read_support_format_raw(struct archive *_a)
{
	struct raw_info *info;
	struct archive_read *a = (struct archive_read *)_a;
	int r;

	archive_check_magic(_a, ARCHIVE_READ_MAGIC,
	    ARCHIVE_STATE_NEW, "archive_read_support_format_raw");

	info = (struct raw_info *)calloc(1, sizeof(*info));
	if (info == NULL) {
		archive_set_error(&a->archive, ENOMEM,
		    "Can't allocate raw_info data");
		return (ARCHIVE_FATAL);
	}

	r = __archive_read_register_format(a,
	    info,
	    "raw",
	    archive_read_format_raw_bid,
	    NULL,
	    archive_read_format_raw_read_header,
	    archive_read_format_raw_read_data,
	    archive_read_format_raw_read_data_skip,
	    NULL,
	    archive_read_format_raw_cleanup,
	    NULL,
	    NULL);
	if (r != ARCHIVE_OK)
		free(info);
	return (r);
}

/*
 * Bid 1 if this is a non-empty file.  Anyone who can really support
 * this should outbid us, so it should generally be safe to use "raw"
 * in conjunction with other formats.  But, this could really confuse
 * folks if there are bid errors or minor file damage, so we don't
 * include "raw" as part of support_format_all().
 */
static int
archive_read_format_raw_bid(struct archive_read *a, int best_bid)
{
	if (best_bid < 1 && __archive_read_ahead(a, 1, NULL) != NULL)
		return (1);
	return (-1);
}

/*
 * Mock up a fake header.
 */
static int
archive_read_format_raw_read_header(struct archive_read *a,
    struct archive_entry *entry)
{
	struct raw_info *info;

	info = (struct raw_info *)(a->format->data);
	if (info->end_of_file)
		return (ARCHIVE_EOF);

	a->archive.archive_format = ARCHIVE_FORMAT_RAW;
	a->archive.archive_format_name = "raw";
	archive_entry_set_pathname(entry, "data");
	archive_entry_set_filetype(entry, AE_IFREG);
	archive_entry_set_perm(entry, 0644);
	/* I'm deliberately leaving most fields unset here. */

	/* Let the filter fill out any fields it might have. */
	return __archive_read_header(a, entry);
}

static int
archive_read_format_raw_read_data(struct archive_read *a,
    const void **buff, size_t *size, int64_t *offset)
{
	struct raw_info *info;
	ssize_t avail;

	info = (struct raw_info *)(a->format->data);

	/* Consume the bytes we read last time. */
	if (info->unconsumed) {
		__archive_read_consume(a, info->unconsumed);
		info->unconsumed = 0;
	}

	if (info->end_of_file)
		return (ARCHIVE_EOF);

	/* Get whatever bytes are immediately available. */
	*buff = __archive_read_ahead(a, 1, &avail);
	if (avail > 0) {
		/* Return the bytes we just read */
		*size = avail;
		*offset = info->offset;
		info->offset += *size;
		info->unconsumed = avail;
		return (ARCHIVE_OK);
	} else if (0 == avail) {
		/* Record and return end-of-file. */
		info->end_of_file = 1;
		*size = 0;
		*offset = info->offset;
		return (ARCHIVE_EOF);
	} else {
		/* Record and return an error. */
		*size = 0;
		*offset = info->offset;
		return ((int)avail);
	}
}

static int
archive_read_format_raw_read_data_skip(struct archive_read *a)
{
	struct raw_info *info = (struct raw_info *)(a->format->data);

	/* Consume the bytes we read last time. */
	if (info->unconsumed) {
		__archive_read_consume(a, info->unconsumed);
		info->unconsumed = 0;
	}
	info->end_of_file = 1;
	return (ARCHIVE_OK);
}

static int
archive_read_format_raw_cleanup(struct archive_read *a)
{
	struct raw_info *info;

	info = (struct raw_info *)(a->format->data);
	free(info);
	a->format->data = NULL;
	return (ARCHIVE_OK);
}
