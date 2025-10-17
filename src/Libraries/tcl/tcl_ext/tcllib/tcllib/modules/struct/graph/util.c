/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#include "tcl.h"
#include <util.h>

static NL* newitem (void* n);


/* Initialize queue data structure.
 */

void
g_nlq_init (NLQ* q)
{
    q->start = q->end = NULL;
}

/* Add item to end of the list
 */

void
g_nlq_append (NLQ* q, void* n)
{
    NL* qi = newitem (n);

    if (!q->end) {
	q->start = q->end = qi;
    } else {
	q->end->next = qi;
	q->end = qi;
    }
}

/* Add item to the front of the list
 */

void
g_nlq_push (NLQ* q, void* n)
{
    NL* qi = newitem (n);

    if (!q->end) {
	q->start = q->end = qi;
    } else {
	qi->next = q->start;
	q->start = qi;
    }
}

/* Return item at front of the list.
 */

void*
g_nlq_pop (NLQ* q)
{
    NL*	  qi = NULL;
    void* n  = NULL;

    if (!q->start) {
	return NULL;
    }

    qi = q->start;
    n  = qi->n;

    q->start = qi->next;
    if (q->end == qi) {
	q->end = NULL;
    }

    ckfree ((char*) qi);
    return n;
}

/* Delete all items in the list.
 */

void*
g_nlq_clear (NLQ* q)
{
    NL* next;
    NL* qi = q->start;

    while (qi) {
	next = qi->next;
	ckfree ((char*) qi);
	qi = next;
    }
    q->start = NULL;
    q->end   = NULL;
}

/* INTERNAL - Create new item to put into the list.
 */

static NL*
newitem (void* n)
{
    NL* qi = (NL*) ckalloc (sizeof (NL));

    qi->n    = n;
    qi->next = NULL;

    return qi;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
