/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "archive.h"
#include "archive_private.h"
#include "archive_read_private.h"
#ifdef __APPLE__
#include "archive_check_entitlement.h"
#endif

static const unsigned char grzip_magic[] = {
	0x47, 0x52, 0x5a, 0x69, 0x70, 0x49, 0x49, 0x00,
	0x02, 0x04, 0x3a, 0x29 };

static int	grzip_bidder_bid(struct archive_read_filter_bidder *,
		    struct archive_read_filter *);
static int	grzip_bidder_init(struct archive_read_filter *);


static const struct archive_read_filter_bidder_vtable
grzip_bidder_vtable = {
	.bid = grzip_bidder_bid,
	.init = grzip_bidder_init,
};

int
archive_read_support_filter_grzip(struct archive *_a)
{
	struct archive_read *a = (struct archive_read *)_a;

#ifdef __APPLE__
	if (!archive_allow_entitlement_filter("grzip")) {
		archive_set_error(_a, ARCHIVE_ERRNO_MISC, "Filter not allow-listed in entitlements");
		return ARCHIVE_FATAL;
	}
#endif

	if (__archive_read_register_bidder(a, NULL, NULL,
				&grzip_bidder_vtable) != ARCHIVE_OK)
		return (ARCHIVE_FATAL);

	/* This filter always uses an external program. */
	archive_set_error(_a, ARCHIVE_ERRNO_MISC,
	    "Using external grzip program for grzip decompression");
	return (ARCHIVE_WARN);
}

/*
 * Bidder just verifies the header and returns the number of verified bits.
 */
static int
grzip_bidder_bid(struct archive_read_filter_bidder *self,
    struct archive_read_filter *filter)
{
	const unsigned char *p;
	ssize_t avail;

	(void)self; /* UNUSED */

	p = __archive_read_filter_ahead(filter, sizeof(grzip_magic), &avail);
	if (p == NULL || avail == 0)
		return (0);

	if (memcmp(p, grzip_magic, sizeof(grzip_magic)))
		return (0);

	return (sizeof(grzip_magic) * 8);
}

static int
grzip_bidder_init(struct archive_read_filter *self)
{
	int r;

	r = __archive_read_program(self, "grzip -d");
	/* Note: We set the format here even if __archive_read_program()
	 * above fails.  We do, after all, know what the format is
	 * even if we weren't able to read it. */
	self->code = ARCHIVE_FILTER_GRZIP;
	self->name = "grzip";
	return (r);
}
