/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

#include "archive.h"
#include "archive_endian.h"
#include "archive_private.h"
#include "archive_read_private.h"
#ifdef __APPLE__
#include "archive_check_entitlement.h"
#endif

struct rpm {
	int64_t		 total_in;
	size_t		 hpos;
	size_t		 hlen;
	unsigned char	 header[16];
	enum {
		ST_LEAD,	/* Skipping 'Lead' section. */
		ST_HEADER,	/* Reading 'Header' section;
				 * first 16 bytes. */
		ST_HEADER_DATA,	/* Skipping 'Header' section. */
		ST_PADDING,	/* Skipping padding data after the
				 * 'Header' section. */
		ST_ARCHIVE	/* Reading 'Archive' section. */
	}		 state;
	int		 first_header;
};
#define RPM_LEAD_SIZE	96	/* Size of 'Lead' section. */

static int	rpm_bidder_bid(struct archive_read_filter_bidder *,
		    struct archive_read_filter *);
static int	rpm_bidder_init(struct archive_read_filter *);

static ssize_t	rpm_filter_read(struct archive_read_filter *,
		    const void **);
static int	rpm_filter_close(struct archive_read_filter *);

#if ARCHIVE_VERSION_NUMBER < 4000000
/* Deprecated; remove in libarchive 4.0 */
int
archive_read_support_compression_rpm(struct archive *a)
{
	return archive_read_support_filter_rpm(a);
}
#endif

static const struct archive_read_filter_bidder_vtable
rpm_bidder_vtable = {
	.bid = rpm_bidder_bid,
	.init = rpm_bidder_init,
};

int
archive_read_support_filter_rpm(struct archive *_a)
{
	struct archive_read *a = (struct archive_read *)_a;

#ifdef __APPLE__
	if (!archive_allow_entitlement_filter("rpm")) {
		archive_set_error(_a, ARCHIVE_ERRNO_MISC, "Format not allow-listed in entitlement");
		return ARCHIVE_FATAL;
	}
#endif

	return __archive_read_register_bidder(a, NULL, "rpm",
			&rpm_bidder_vtable);
}

static int
rpm_bidder_bid(struct archive_read_filter_bidder *self,
    struct archive_read_filter *filter)
{
	const unsigned char *b;
	ssize_t avail;
	int bits_checked;

	(void)self; /* UNUSED */

	b = __archive_read_filter_ahead(filter, 8, &avail);
	if (b == NULL)
		return (0);

	bits_checked = 0;
	/*
	 * Verify Header Magic Bytes : 0XED 0XAB 0XEE 0XDB
	 */
	if (memcmp(b, "\xED\xAB\xEE\xDB", 4) != 0)
		return (0);
	bits_checked += 32;
	/*
	 * Check major version.
	 */
	if (b[4] != 3 && b[4] != 4)
		return (0);
	bits_checked += 8;
	/*
	 * Check package type; binary or source.
	 */
	if (b[6] != 0)
		return (0);
	bits_checked += 8;
	if (b[7] != 0 && b[7] != 1)
		return (0);
	bits_checked += 8;

	return (bits_checked);
}

static const struct archive_read_filter_vtable
rpm_reader_vtable = {
	.read = rpm_filter_read,
	.close = rpm_filter_close,
};

static int
rpm_bidder_init(struct archive_read_filter *self)
{
	struct rpm   *rpm;

	self->code = ARCHIVE_FILTER_RPM;
	self->name = "rpm";

	rpm = (struct rpm *)calloc(1, sizeof(*rpm));
	if (rpm == NULL) {
		archive_set_error(&self->archive->archive, ENOMEM,
		    "Can't allocate data for rpm");
		return (ARCHIVE_FATAL);
	}

	self->data = rpm;
	rpm->state = ST_LEAD;
	self->vtable = &rpm_reader_vtable;

	return (ARCHIVE_OK);
}

static ssize_t
rpm_filter_read(struct archive_read_filter *self, const void **buff)
{
	struct rpm *rpm;
	const unsigned char *b;
	ssize_t avail_in, total;
	size_t used, n;
	uint32_t section;
	uint32_t bytes;

	rpm = (struct rpm *)self->data;
	*buff = NULL;
	total = avail_in = 0;
	b = NULL;
	used = 0;
	do {
		if (b == NULL) {
			b = __archive_read_filter_ahead(self->upstream, 1,
			    &avail_in);
			if (b == NULL) {
				if (avail_in < 0)
					return (ARCHIVE_FATAL);
				else
					break;
			}
		}

		switch (rpm->state) {
		case ST_LEAD:
			if (rpm->total_in + avail_in < RPM_LEAD_SIZE)
				used += avail_in;
			else {
				n = (size_t)(RPM_LEAD_SIZE - rpm->total_in);
				used += n;
				b += n;
				rpm->state = ST_HEADER;
				rpm->hpos = 0;
				rpm->hlen = 0;
				rpm->first_header = 1;
			}
			break;
		case ST_HEADER:
			n = 16 - rpm->hpos;
			if (n > avail_in - used)
				n = avail_in - used;
			memcpy(rpm->header+rpm->hpos, b, n);
			b += n;
			used += n;
			rpm->hpos += n;

			if (rpm->hpos == 16) {
				if (rpm->header[0] != 0x8e ||
				    rpm->header[1] != 0xad ||
				    rpm->header[2] != 0xe8 ||
				    rpm->header[3] != 0x01) {
					if (rpm->first_header) {
						archive_set_error(
						    &self->archive->archive,
						    ARCHIVE_ERRNO_FILE_FORMAT,
						    "Unrecognized rpm header");
						return (ARCHIVE_FATAL);
					}
					rpm->state = ST_ARCHIVE;
					*buff = rpm->header;
					total = rpm->hpos;
					break;
				}
				/* Calculate 'Header' length. */
				section = archive_be32dec(rpm->header+8);
				bytes = archive_be32dec(rpm->header+12);
				rpm->hlen = 16 + section * 16 + bytes;
				rpm->state = ST_HEADER_DATA;
				rpm->first_header = 0;
			}
			break;
		case ST_HEADER_DATA:
			n = rpm->hlen - rpm->hpos;
			if (n > avail_in - used)
				n = avail_in - used;
			b += n;
			used += n;
			rpm->hpos += n;
			if (rpm->hpos == rpm->hlen)
				rpm->state = ST_PADDING;
			break;
		case ST_PADDING:
			while (used < (size_t)avail_in) {
				if (*b != 0) {
					/* Read next header. */
					rpm->state = ST_HEADER;
					rpm->hpos = 0;
					rpm->hlen = 0;
					break;
				}
				b++;
				used++;
			}
			break;
		case ST_ARCHIVE:
			*buff = b;
			total = avail_in;
			used = avail_in;
			break;
		}
		if (used == (size_t)avail_in) {
			rpm->total_in += used;
			__archive_read_filter_consume(self->upstream, used);
			b = NULL;
			used = 0;
		}
	} while (total == 0 && avail_in > 0);

	if (used > 0 && b != NULL) {
		rpm->total_in += used;
		__archive_read_filter_consume(self->upstream, used);
	}
	return (total);
}

static int
rpm_filter_close(struct archive_read_filter *self)
{
	struct rpm *rpm;

	rpm = (struct rpm *)self->data;
	free(rpm);

	return (ARCHIVE_OK);
}


