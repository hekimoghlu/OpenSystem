/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#include <stdbool.h>
#include <stddef.h>

/*
 * Thread unsafe interface: use a single process-wide hash table and
 * forward calls to *_r() functions.
 */

static struct hsearch_data global_hashtable;
static bool global_hashtable_initialized = false;

int
hcreate(size_t nel)
{

	return (1);
}

void
hdestroy(void)
{

	/* Destroy global hash table if present. */
	if (global_hashtable_initialized) {
		hdestroy_r(&global_hashtable);
		global_hashtable_initialized = false;
	}
}

ENTRY *
hsearch(ENTRY item, ACTION action)
{
	ENTRY *retval;

	/* Create global hash table if needed. */
	if (!global_hashtable_initialized) {
		if (hcreate_r(0, &global_hashtable) == 0)
			return (NULL);
		global_hashtable_initialized = true;
	}
	if (hsearch_r(item, action, &retval, &global_hashtable) == 0)
		return (NULL);
	return (retval);
}
