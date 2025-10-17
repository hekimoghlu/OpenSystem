/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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
#include "archive_entry.h"
#include "archive_private.h"
#include "archive_read_private.h"

#ifdef HAVE_MAC_QUARANTINE
#include "archive_write_disk_private.h"

#include <quarantine.h>
#endif

int
archive_read_extract(struct archive *_a, struct archive_entry *entry, int flags)
{
	struct archive_read_extract *extract;
	struct archive_read * a = (struct archive_read *)_a;

	extract = __archive_read_get_extract(a);
	if (extract == NULL)
		return (ARCHIVE_FATAL);

	/* If we haven't initialized the archive_write_disk object, do it now. */
	if (extract->ad == NULL) {
		extract->ad = archive_write_disk_new();
		if (extract->ad == NULL) {
			archive_set_error(&a->archive, ENOMEM, "Can't extract");
			return (ARCHIVE_FATAL);
		}
		archive_write_disk_set_standard_lookup(extract->ad);

#ifdef HAVE_MAC_QUARANTINE
		/* propagate the quarantine flag */

		if (a->qf) {
			archive_write_disk_set_quarantine(extract->ad, a->qf);
		}
#endif // HAVE_MAC_QUARANTINE

	}


	archive_write_disk_set_options(extract->ad, flags);
	return (archive_read_extract2(&a->archive, entry, extract->ad));
}
