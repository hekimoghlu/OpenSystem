/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

int
archive_read_support_filter_by_code(struct archive *a, int filter_code)
{
	archive_check_magic(a, ARCHIVE_READ_MAGIC,
	    ARCHIVE_STATE_NEW, "archive_read_support_filter_by_code");

	switch (filter_code) {
	case ARCHIVE_FILTER_NONE:
		return archive_read_support_filter_none(a);
		break;
	case ARCHIVE_FILTER_GZIP:
		return archive_read_support_filter_gzip(a);
		break;
	case ARCHIVE_FILTER_BZIP2:
		return archive_read_support_filter_bzip2(a);
		break;
	case ARCHIVE_FILTER_COMPRESS:
		return archive_read_support_filter_compress(a);
		break;
	case ARCHIVE_FILTER_LZMA:
		return archive_read_support_filter_lzma(a);
		break;
	case ARCHIVE_FILTER_XZ:
		return archive_read_support_filter_xz(a);
		break;
	case ARCHIVE_FILTER_UU:
		return archive_read_support_filter_uu(a);
		break;
	case ARCHIVE_FILTER_RPM:
		return archive_read_support_filter_rpm(a);
		break;
	case ARCHIVE_FILTER_LZIP:
		return archive_read_support_filter_lzip(a);
		break;
	case ARCHIVE_FILTER_LRZIP:
		return archive_read_support_filter_lrzip(a);
		break;
	case ARCHIVE_FILTER_LZOP:
		return archive_read_support_filter_lzop(a);
		break;
	case ARCHIVE_FILTER_GRZIP:
		return archive_read_support_filter_grzip(a);
		break;
	case ARCHIVE_FILTER_LZ4:
		return archive_read_support_filter_lz4(a);
		break;
	case ARCHIVE_FILTER_ZSTD:
		return archive_read_support_filter_zstd(a);
		break;
	}
	return (ARCHIVE_FATAL);
}
