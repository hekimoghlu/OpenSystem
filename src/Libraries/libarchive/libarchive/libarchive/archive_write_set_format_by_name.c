/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
	{ "7zip",	archive_write_set_format_7zip },
	{ "ar",		archive_write_set_format_ar_bsd },
	{ "arbsd",	archive_write_set_format_ar_bsd },
	{ "argnu",	archive_write_set_format_ar_svr4 },
	{ "arsvr4",	archive_write_set_format_ar_svr4 },
	{ "bin",	archive_write_set_format_cpio_bin },
	{ "bsdtar",	archive_write_set_format_pax_restricted },
	{ "cd9660",	archive_write_set_format_iso9660 },
	{ "cpio",	archive_write_set_format_cpio },
	{ "gnutar",	archive_write_set_format_gnutar },
	{ "iso",	archive_write_set_format_iso9660 },
	{ "iso9660",	archive_write_set_format_iso9660 },
	{ "mtree",	archive_write_set_format_mtree },
	{ "mtree-classic",	archive_write_set_format_mtree_classic },
	{ "newc",	archive_write_set_format_cpio_newc },
	{ "odc",	archive_write_set_format_cpio_odc },
	{ "oldtar",	archive_write_set_format_v7tar },
	{ "pax",	archive_write_set_format_pax },
	{ "paxr",	archive_write_set_format_pax_restricted },
	{ "posix",	archive_write_set_format_pax },
	{ "pwb",	archive_write_set_format_cpio_pwb },
	{ "raw",	archive_write_set_format_raw },
	{ "rpax",	archive_write_set_format_pax_restricted },
	{ "shar",	archive_write_set_format_shar },
	{ "shardump",	archive_write_set_format_shar_dump },
	{ "ustar",	archive_write_set_format_ustar },
	{ "v7tar",	archive_write_set_format_v7tar },
	{ "v7",		archive_write_set_format_v7tar },
	{ "warc",	archive_write_set_format_warc },
	{ "xar",	archive_write_set_format_xar },
	{ "zip",	archive_write_set_format_zip },
	{ NULL,		NULL }
};

int
archive_write_set_format_by_name(struct archive *a, const char *name)
{
	int i;

	for (i = 0; names[i].name != NULL; i++) {
		if (strcmp(name, names[i].name) == 0)
			return ((names[i].setter)(a));
	}

	archive_set_error(a, EINVAL, "No such format '%s'", name);
	a->state = ARCHIVE_STATE_FATAL;
	return (ARCHIVE_FATAL);
}
