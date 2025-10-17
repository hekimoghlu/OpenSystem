/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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

#include <search.h>
#include <stdlib.h>

#include "hsearch.h"

int
hcreate_r(size_t nel, struct hsearch_data *htab)
{
	struct __hsearch *hsearch;

	/*
	 * Allocate a hash table object. Ignore the provided hint and start
	 * off with a table of sixteen entries. In most cases this hint is
	 * just a wild guess. Resizing the table dynamically if the use
	 * increases a threshold does not affect the worst-case running time.
	 */
	hsearch = malloc(sizeof(*hsearch));
	if (hsearch == NULL)
		return 0;
	hsearch->entries = calloc(16, sizeof(ENTRY));
	if (hsearch->entries == NULL) {
		free(hsearch);
		return 0;
	}

	/*
	 * Pick a random initialization for the FNV-1a hashing. This makes it
	 * hard to come up with a fixed set of keys to force hash collisions.
	 */
	arc4random_buf(&hsearch->offset_basis, sizeof(hsearch->offset_basis));
	hsearch->index_mask = 0xf;
	hsearch->entries_used = 0;
	htab->__hsearch = hsearch;
	return 1;
}
