/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "archive.h"
#include "archive_private.h"

/* A table that maps filter codes to functions. */
static const
struct { int code; int (*setter)(struct archive *); } codes[] =
{
	{ ARCHIVE_FILTER_NONE,		archive_write_add_filter_none },
	{ ARCHIVE_FILTER_GZIP,		archive_write_add_filter_gzip },
	{ ARCHIVE_FILTER_BZIP2,		archive_write_add_filter_bzip2 },
	{ ARCHIVE_FILTER_COMPRESS,	archive_write_add_filter_compress },
	{ ARCHIVE_FILTER_GRZIP,		archive_write_add_filter_grzip },
	{ ARCHIVE_FILTER_LRZIP,		archive_write_add_filter_lrzip },
	{ ARCHIVE_FILTER_LZ4,		archive_write_add_filter_lz4 },
	{ ARCHIVE_FILTER_LZIP,		archive_write_add_filter_lzip },
	{ ARCHIVE_FILTER_LZMA,		archive_write_add_filter_lzma },
	{ ARCHIVE_FILTER_LZOP,		archive_write_add_filter_lzip },
	{ ARCHIVE_FILTER_UU,		archive_write_add_filter_uuencode },
	{ ARCHIVE_FILTER_XZ,		archive_write_add_filter_xz },
	{ ARCHIVE_FILTER_ZSTD,		archive_write_add_filter_zstd },
	{ -1,			NULL }
};

int
archive_write_add_filter(struct archive *a, int code)
{
	int i;

	for (i = 0; codes[i].code != -1; i++) {
		if (code == codes[i].code)
			return ((codes[i].setter)(a));
	}

	archive_set_error(a, EINVAL, "No such filter");
	return (ARCHIVE_FATAL);
}
