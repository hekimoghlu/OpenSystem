/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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

#define LRZIP_HEADER_MAGIC "LRZI"
#define LRZIP_HEADER_MAGIC_LEN 4

static int	lrzip_bidder_bid(struct archive_read_filter_bidder *,
		    struct archive_read_filter *);
static int	lrzip_bidder_init(struct archive_read_filter *);


static const struct archive_read_filter_bidder_vtable
lrzip_bidder_vtable = {
	.bid = lrzip_bidder_bid,
	.init = lrzip_bidder_init,
};

int
archive_read_support_filter_lrzip(struct archive *_a)
{
	struct archive_read *a = (struct archive_read *)_a;

#ifdef __APPLE__
	if (!archive_allow_entitlement_filter("lrzip")) {
		archive_set_error(_a, ARCHIVE_ERRNO_MISC, "Format not allow-listed in entitlements");
		return ARCHIVE_FATAL;
	}
#endif

	if (__archive_read_register_bidder(a, NULL, "lrzip",
				&lrzip_bidder_vtable) != ARCHIVE_OK)
		return (ARCHIVE_FATAL);

	/* This filter always uses an external program. */
	archive_set_error(_a, ARCHIVE_ERRNO_MISC,
	    "Using external lrzip program for lrzip decompression");
	return (ARCHIVE_WARN);
}

/*
 * Bidder just verifies the header and returns the number of verified bits.
 */
static int
lrzip_bidder_bid(struct archive_read_filter_bidder *self,
    struct archive_read_filter *filter)
{
	const unsigned char *p;
	ssize_t avail, len;
	int i;

	(void)self; /* UNUSED */
	/* Start by looking at the first six bytes of the header, which
	 * is all fixed layout. */
	len = 6;
	p = __archive_read_filter_ahead(filter, len, &avail);
	if (p == NULL || avail == 0)
		return (0);

	if (memcmp(p, LRZIP_HEADER_MAGIC, LRZIP_HEADER_MAGIC_LEN))
		return (0);

	/* current major version is always 0, verify this */
	if (p[LRZIP_HEADER_MAGIC_LEN])
		return 0;
	/* support only v0.6+ lrzip for sanity */
	i = p[LRZIP_HEADER_MAGIC_LEN + 1];
	if ((i < 6) || (i > 10))
		return 0;

	return (int)len;
}

static int
lrzip_bidder_init(struct archive_read_filter *self)
{
	int r;

	r = __archive_read_program(self, "lrzip -d -q");
	/* Note: We set the format here even if __archive_read_program()
	 * above fails.  We do, after all, know what the format is
	 * even if we weren't able to read it. */
	self->code = ARCHIVE_FILTER_LRZIP;
	self->name = "lrzip";
	return (r);
}
