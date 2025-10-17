/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <errno.h>
#include <limits.h>
#include <search.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hsearch.h"

/*
 * Look up an unused entry in the hash table for a given hash. For this
 * implementation we use quadratic probing. Quadratic probing has the
 * advantage of preventing primary clustering.
 */
static ENTRY *
hsearch_lookup_free(struct __hsearch *hsearch, size_t hash)
{
	size_t index, i;

	for (index = hash, i = 0;; index += ++i) {
		ENTRY *entry = &hsearch->entries[index & hsearch->index_mask];
		if (entry->key == NULL)
			return (entry);
	}
}

/*
 * Computes an FNV-1a hash of the key. Depending on the pointer size, this
 * either uses the 32- or 64-bit FNV prime.
 */
static size_t
hsearch_hash(size_t offset_basis, const char *str)
{
	size_t hash;

	hash = offset_basis;
	while (*str != '\0') {
		hash ^= (uint8_t)*str++;
		if (sizeof(size_t) * CHAR_BIT <= 32)
			hash *= UINT32_C(16777619);
		else
			hash *= UINT64_C(1099511628211);
	}
	return (hash);
}

int
hsearch_r(ENTRY item, ACTION action, ENTRY **retval, struct hsearch_data *htab)
{
	struct __hsearch *hsearch;
	ENTRY *entry, *old_entries, *new_entries;
	size_t hash, index, i, old_hash, old_count, new_count;

	hsearch = htab->__hsearch;
	hash = hsearch_hash(hsearch->offset_basis, item.key);

	/*
	 * Search the hash table for an existing entry for this key.
	 * Stop searching if we run into an unused hash table entry.
	 */
	for (index = hash, i = 0;; index += ++i) {
		entry = &hsearch->entries[index & hsearch->index_mask];
		if (entry->key == NULL)
			break;
		if (strcmp(entry->key, item.key) == 0) {
			*retval = entry;
			return (1);
		}
	}

	/* Only perform the insertion if action is set to ENTER. */
	if (action == FIND) {
		errno = ESRCH;
		return (0);
	}

	if (hsearch->entries_used * 2 >= hsearch->index_mask) {
		/* Preserve the old hash table entries. */
		old_count = hsearch->index_mask + 1;
		old_entries = hsearch->entries;

		/*
		 * Allocate and install a new table if insertion would
		 * yield a hash table that is more than 50% used. By
		 * using 50% as a threshold, a lookup will only take up
		 * to two steps on average.
		 */
		new_count = (hsearch->index_mask + 1) * 2;
		new_entries = calloc(new_count, sizeof(ENTRY));
		if (new_entries == NULL)
			return (0);
		hsearch->entries = new_entries;
		hsearch->index_mask = new_count - 1;

		/* Copy over the entries from the old table to the new table. */
		for (i = 0; i < old_count; ++i) {
			entry = &old_entries[i];
			if (entry->key != NULL) {
				old_hash = hsearch_hash(hsearch->offset_basis,
				    entry->key);
				*hsearch_lookup_free(hsearch, old_hash) =
				    *entry;
			}
		}

		/* Destroy the old hash table entries. */
		free(old_entries);

		/*
		 * Perform a new lookup for a free table entry, so that
		 * we insert the entry into the new hash table.
		 */
		hsearch = htab->__hsearch;
		entry = hsearch_lookup_free(hsearch, hash);
	}

	/* Insert the new entry into the hash table. */
	*entry = item;
	++hsearch->entries_used;
	*retval = entry;
	return (1);
}
