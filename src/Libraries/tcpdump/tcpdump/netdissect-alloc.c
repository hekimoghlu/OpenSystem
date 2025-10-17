/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include "netdissect-alloc.h"

/*
 * nd_free_all() is intended to be used after a packet printing
 */

/* Add a memory chunk in allocation linked list */
void
nd_add_alloc_list(netdissect_options *ndo, nd_mem_chunk_t *chunkp)
{
	if (ndo->ndo_last_mem_p == NULL)	/* first memory allocation */
		chunkp->prev_mem_p = NULL;
	else					/* previous memory allocation */
		chunkp->prev_mem_p = ndo->ndo_last_mem_p;
	ndo->ndo_last_mem_p = chunkp;
}

/* malloc replacement, with tracking in a linked list */
void *
nd_malloc(netdissect_options *ndo, size_t size)
{
	nd_mem_chunk_t *chunkp = malloc(sizeof(nd_mem_chunk_t) + size);
	if (chunkp == NULL)
		return NULL;
	nd_add_alloc_list(ndo, chunkp);
	return chunkp + 1;
}

/* Free chunks in allocation linked list from last to first */
void
nd_free_all(netdissect_options *ndo)
{
	nd_mem_chunk_t *current, *previous;
	current = ndo->ndo_last_mem_p;
	while (current != NULL) {
		previous = current->prev_mem_p;
		free(current);
		current = previous;
	}
	ndo->ndo_last_mem_p = NULL;
}
