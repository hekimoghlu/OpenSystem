/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif
#ifdef HAVE_LINUX_FS_H
#include <linux/fs.h>	/* for Linux file flags */
#endif
/*
 * Some Linux distributions have both linux/ext2_fs.h and ext2fs/ext2_fs.h.
 * As the include guards don't agree, the order of include is important.
 */
#ifdef HAVE_LINUX_EXT2_FS_H
#include <linux/ext2_fs.h>	/* for Linux file flags */
#endif
#if defined(HAVE_EXT2FS_EXT2_FS_H) && !defined(__CYGWIN__)
#include <ext2fs/ext2_fs.h>	/* for Linux file flags */
#endif
#include <stddef.h>
#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_WCHAR_H
#include <wchar.h>
#endif

#include "archive.h"
#include "archive_entry.h"
#include "archive_private.h"
#include "archive_entry_private.h"

/*
 * extended attribute handling
 */

void
archive_entry_xattr_clear(struct archive_entry *entry)
{
	struct ae_xattr	*xp;

	while (entry->xattr_head != NULL) {
		xp = entry->xattr_head->next;
		free(entry->xattr_head->name);
		free(entry->xattr_head->value);
		free(entry->xattr_head);
		entry->xattr_head = xp;
	}

	entry->xattr_head = NULL;
}

void
archive_entry_xattr_add_entry(struct archive_entry *entry,
	const char *name, const void *value, size_t size)
{
	struct ae_xattr	*xp;

	if ((xp = (struct ae_xattr *)malloc(sizeof(struct ae_xattr))) == NULL)
		__archive_errx(1, "Out of memory");

	if ((xp->name = strdup(name)) == NULL)
		__archive_errx(1, "Out of memory");

	if ((xp->value = malloc(size)) != NULL) {
		memcpy(xp->value, value, size);
		xp->size = size;
	} else
		xp->size = 0;

	xp->next = entry->xattr_head;
	entry->xattr_head = xp;
}


/*
 * returns number of the extended attribute entries
 */
int
archive_entry_xattr_count(struct archive_entry *entry)
{
	struct ae_xattr *xp;
	int count = 0;

	for (xp = entry->xattr_head; xp != NULL; xp = xp->next)
		count++;

	return count;
}

int
archive_entry_xattr_reset(struct archive_entry * entry)
{
	entry->xattr_p = entry->xattr_head;

	return archive_entry_xattr_count(entry);
}

int
archive_entry_xattr_next(struct archive_entry * entry,
	const char **name, const void **value, size_t *size)
{
	if (entry->xattr_p) {
		*name = entry->xattr_p->name;
		*value = entry->xattr_p->value;
		*size = entry->xattr_p->size;

		entry->xattr_p = entry->xattr_p->next;

		return (ARCHIVE_OK);
	} else {
		*name = NULL;
		*value = NULL;
		*size = (size_t)0;
		return (ARCHIVE_WARN);
	}
}

/*
 * end of xattr handling
 */
