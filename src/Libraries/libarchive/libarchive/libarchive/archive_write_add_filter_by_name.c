/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "archive.h"
#include "archive_private.h"

/* A table that maps names to functions. */
static const
struct { const char *name; int (*setter)(struct archive *); } names[] =
{
	{ "b64encode",		archive_write_add_filter_b64encode },
	{ "bzip2",		archive_write_add_filter_bzip2 },
	{ "compress",		archive_write_add_filter_compress },
	{ "grzip",		archive_write_add_filter_grzip },
	{ "gzip",		archive_write_add_filter_gzip },
	{ "lrzip",		archive_write_add_filter_lrzip },
	{ "lz4",		archive_write_add_filter_lz4 },
	{ "lzip",		archive_write_add_filter_lzip },
	{ "lzma",		archive_write_add_filter_lzma },
	{ "lzop",		archive_write_add_filter_lzop },
	{ "uuencode",		archive_write_add_filter_uuencode },
	{ "xz",			archive_write_add_filter_xz },
	{ "zstd",		archive_write_add_filter_zstd },
	{ NULL,			NULL }
};

int
archive_write_add_filter_by_name(struct archive *a, const char *name)
{
	int i;

	for (i = 0; names[i].name != NULL; i++) {
		if (strcmp(name, names[i].name) == 0)
			return ((names[i].setter)(a));
	}

	archive_set_error(a, EINVAL, "No such filter '%s'", name);
	a->state = ARCHIVE_STATE_FATAL;
	return (ARCHIVE_FATAL);
}
