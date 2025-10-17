/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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

#include "archive.h"
#include "archive_private.h"

int
archive_read_support_format_by_code(struct archive *a, int format_code)
{
	archive_check_magic(a, ARCHIVE_READ_MAGIC,
	    ARCHIVE_STATE_NEW, "archive_read_support_format_by_code");

	switch (format_code & ARCHIVE_FORMAT_BASE_MASK) {
	case ARCHIVE_FORMAT_7ZIP:
		return archive_read_support_format_7zip(a);
		break;
	case ARCHIVE_FORMAT_AR:
		return archive_read_support_format_ar(a);
		break;
	case ARCHIVE_FORMAT_CAB:
		return archive_read_support_format_cab(a);
		break;
	case ARCHIVE_FORMAT_CPIO:
		return archive_read_support_format_cpio(a);
		break;
	case ARCHIVE_FORMAT_EMPTY:
		return archive_read_support_format_empty(a);
		break;
	case ARCHIVE_FORMAT_ISO9660:
		return archive_read_support_format_iso9660(a);
		break;
	case ARCHIVE_FORMAT_LHA:
		return archive_read_support_format_lha(a);
		break;
	case ARCHIVE_FORMAT_MTREE:
		return archive_read_support_format_mtree(a);
		break;
	case ARCHIVE_FORMAT_RAR:
		return archive_read_support_format_rar(a);
		break;
	case ARCHIVE_FORMAT_RAR_V5:
		return archive_read_support_format_rar5(a);
		break;
	case ARCHIVE_FORMAT_RAW:
		return archive_read_support_format_raw(a);
		break;
	case ARCHIVE_FORMAT_TAR:
		return archive_read_support_format_tar(a);
		break;
	case ARCHIVE_FORMAT_WARC:
		return archive_read_support_format_warc(a);
		break;
	case ARCHIVE_FORMAT_XAR:
		return archive_read_support_format_xar(a);
		break;
	case ARCHIVE_FORMAT_ZIP:
		return archive_read_support_format_zip(a);
		break;
	}
	archive_set_error(a, ARCHIVE_ERRNO_PROGRAMMER,
	    "Invalid format code specified");
	return (ARCHIVE_FATAL);
}
