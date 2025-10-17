/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#include "archive_entry.h"
#include "archive_private.h"
#include "archive_entry_private.h"

/*
 * sparse handling
 */

void
archive_entry_sparse_clear(struct archive_entry *entry)
{
	struct ae_sparse *sp;

	while (entry->sparse_head != NULL) {
		sp = entry->sparse_head->next;
		free(entry->sparse_head);
		entry->sparse_head = sp;
	}
	entry->sparse_tail = NULL;
}

void
archive_entry_sparse_add_entry(struct archive_entry *entry,
	la_int64_t offset, la_int64_t length)
{
	struct ae_sparse *sp;

	if (offset < 0 || length < 0)
		/* Invalid value */
		return;
	if (offset > INT64_MAX - length ||
	    offset + length > archive_entry_size(entry))
		/* A value of "length" parameter is too large. */
		return;
	if ((sp = entry->sparse_tail) != NULL) {
		if (sp->offset + sp->length > offset)
			/* Invalid value. */
			return;
		if (sp->offset + sp->length == offset) {
			if (sp->offset + sp->length + length < 0)
				/* A value of "length" parameter is
				 * too large. */
				return;
			/* Expand existing sparse block size. */
			sp->length += length;
			return;
		}
	}

	if ((sp = (struct ae_sparse *)malloc(sizeof(*sp))) == NULL)
		/* XXX Error XXX */
		return;

	sp->offset = offset;
	sp->length = length;
	sp->next = NULL;

	if (entry->sparse_head == NULL)
		entry->sparse_head = entry->sparse_tail = sp;
	else {
		/* Add a new sparse block to the tail of list. */
		if (entry->sparse_tail != NULL)
			entry->sparse_tail->next = sp;
		entry->sparse_tail = sp;
	}
}


/*
 * returns number of the sparse entries
 */
int
archive_entry_sparse_count(struct archive_entry *entry)
{
	struct ae_sparse *sp;
	int count = 0;

	for (sp = entry->sparse_head; sp != NULL; sp = sp->next)
		count++;

	/*
	 * Sanity check if this entry is exactly sparse.
	 * If amount of sparse blocks is just one and it indicates the whole
	 * file data, we should remove it and return zero.
	 */
	if (count == 1) {
		sp = entry->sparse_head;
		if (sp->offset == 0 &&
		    sp->length >= archive_entry_size(entry)) {
			count = 0;
			archive_entry_sparse_clear(entry);
		}
	}

	return (count);
}

int
archive_entry_sparse_reset(struct archive_entry * entry)
{
	entry->sparse_p = entry->sparse_head;

	return archive_entry_sparse_count(entry);
}

int
archive_entry_sparse_next(struct archive_entry * entry,
	la_int64_t *offset, la_int64_t *length)
{
	if (entry->sparse_p) {
		*offset = entry->sparse_p->offset;
		*length = entry->sparse_p->length;

		entry->sparse_p = entry->sparse_p->next;

		return (ARCHIVE_OK);
	} else {
		*offset = 0;
		*length = 0;
		return (ARCHIVE_WARN);
	}
}

/*
 * end of sparse handling
 */
