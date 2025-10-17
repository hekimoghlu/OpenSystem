/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include "archive_read_private.h"
#ifdef __APPLE__
#include "archive_check_entitlement.h"
#endif


static int	archive_read_format_empty_bid(struct archive_read *, int);
static int	archive_read_format_empty_read_data(struct archive_read *,
		    const void **, size_t *, int64_t *);
static int	archive_read_format_empty_read_header(struct archive_read *,
		    struct archive_entry *);
int
archive_read_support_format_empty(struct archive *_a)
{
	struct archive_read *a = (struct archive_read *)_a;
	int r;

#ifdef __APPLE__
	if (!archive_allow_entitlement_format("empty")) {
		archive_set_error(&a->archive, ARCHIVE_ERRNO_MISC,
						  "Format not allow-listed in entitlements");
		return ARCHIVE_FATAL;
	}
#endif

	archive_check_magic(_a, ARCHIVE_READ_MAGIC,
	    ARCHIVE_STATE_NEW, "archive_read_support_format_empty");

	r = __archive_read_register_format(a,
	    NULL,
	    "empty",
	    archive_read_format_empty_bid,
	    NULL,
	    archive_read_format_empty_read_header,
	    archive_read_format_empty_read_data,
	    NULL,
	    NULL,
	    NULL,
	    NULL,
	    NULL);

	return (r);
}


static int
archive_read_format_empty_bid(struct archive_read *a, int best_bid)
{
	if (best_bid < 1 && __archive_read_ahead(a, 1, NULL) == NULL)
		return (1);
	return (-1);
}

static int
archive_read_format_empty_read_header(struct archive_read *a,
    struct archive_entry *entry)
{
	(void)a; /* UNUSED */
	(void)entry; /* UNUSED */

	a->archive.archive_format = ARCHIVE_FORMAT_EMPTY;
	a->archive.archive_format_name = "Empty file";

	return (ARCHIVE_EOF);
}

static int
archive_read_format_empty_read_data(struct archive_read *a,
    const void **buff, size_t *size, int64_t *offset)
{
	(void)a; /* UNUSED */
	(void)buff; /* UNUSED */
	(void)size; /* UNUSED */
	(void)offset; /* UNUSED */

	return (ARCHIVE_EOF);
}
