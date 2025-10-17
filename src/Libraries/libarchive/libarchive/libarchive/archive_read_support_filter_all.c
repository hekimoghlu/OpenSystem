/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#include "archive_private.h"

#if ARCHIVE_VERSION_NUMBER < 4000000
/* Deprecated; remove in libarchive 4.0 */
int
archive_read_support_compression_all(struct archive *a)
{
	return archive_read_support_filter_all(a);
}
#endif

int
archive_read_support_filter_all(struct archive *a)
{
	archive_check_magic(a, ARCHIVE_READ_MAGIC,
	    ARCHIVE_STATE_NEW, "archive_read_support_filter_all");

	/* Bzip falls back to "bunzip2" command-line */
	archive_read_support_filter_bzip2(a);
	/* The decompress code doesn't use an outside library. */
	archive_read_support_filter_compress(a);
	/* Gzip decompress falls back to "gzip -d" command-line. */
	archive_read_support_filter_gzip(a);
	/* Lzip falls back to "unlzip" command-line program. */
	archive_read_support_filter_lzip(a);
	/* The LZMA file format has a very weak signature, so it
	 * may not be feasible to keep this here, but we'll try.
	 * This will come back out if there are problems. */
	/* Lzma falls back to "unlzma" command-line program. */
	archive_read_support_filter_lzma(a);
	/* Xz falls back to "unxz" command-line program. */
	archive_read_support_filter_xz(a);
	/* The decode code doesn't use an outside library. */
	archive_read_support_filter_uu(a);
	/* The decode code doesn't use an outside library. */
	archive_read_support_filter_rpm(a);
	/* The decode code always uses "lrzip -q -d" command-line. */
	archive_read_support_filter_lrzip(a);
	/* Lzop decompress falls back to "lzop -d" command-line. */
	archive_read_support_filter_lzop(a);
	/* The decode code always uses "grzip -d" command-line. */
	archive_read_support_filter_grzip(a);
	/* Lz4 falls back to "lz4 -d" command-line program. */
	archive_read_support_filter_lz4(a);
	/* Zstd falls back to "zstd -d" command-line program. */
	archive_read_support_filter_zstd(a);

	/* Note: We always return ARCHIVE_OK here, even if some of the
	 * above return ARCHIVE_WARN.  The intent here is to enable
	 * "as much as possible."  Clients who need specific
	 * compression should enable those individually so they can
	 * verify the level of support. */
	/* Clear any warning messages set by the above functions. */
	archive_clear_error(a);
	return (ARCHIVE_OK);
}
