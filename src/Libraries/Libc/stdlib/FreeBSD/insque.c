/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
__FBSDID("$FreeBSD: src/lib/libc/stdlib/insque.c,v 1.3 2003/01/04 07:34:41 tjr Exp $");

#define	_SEARCH_PRIVATE
#include <search.h>
#ifdef DEBUG
#include <stdio.h>
#else
#include <stdlib.h>	/* for NULL */
#endif

void
insque(void *element, void *pred)
{
	struct que_elem *prev, *next, *elem;

	elem = (struct que_elem *)element;
	prev = (struct que_elem *)pred;

	if (prev == NULL) {
		elem->prev = elem->next = NULL;
		return;
	}

	next = prev->next;
	if (next != NULL) {
#ifdef DEBUG
		if (next->prev != prev) {
			fprintf(stderr, "insque: Inconsistency detected:"
			    " next(%p)->prev(%p) != prev(%p)\n",
			    next, next->prev, prev);
		}
#endif
		next->prev = elem;
	}
	prev->next = elem;
	elem->prev = prev;
	elem->next = next;
}
