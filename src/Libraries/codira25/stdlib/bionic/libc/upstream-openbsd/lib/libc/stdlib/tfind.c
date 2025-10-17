/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
/*
 * Tree search generalized from Knuth (6.2.2) Algorithm T just like
 * the AT&T man page says.
 *
 * The node_t structure is for internal use only
 *
 * Written by reading the System V Interface Definition, not the code.
 *
 * Totally public domain.
 */
#include <search.h>

typedef struct node_t
{
    char	  *key;
    struct node_t *llink, *rlink;
} node;

/* find a node, or return 0 */
void *
tfind(const void *vkey, void * const *vrootp,
    int (*compar)(const void *, const void *))
{
    char *key = (char *)vkey;
    node **rootp = (node **)vrootp;

    if (rootp == (struct node_t **)0)
	return ((struct node_t *)0);
    while (*rootp != (struct node_t *)0) {	/* T1: */
	int r;
	if ((r = (*compar)(key, (*rootp)->key)) == 0)	/* T2: */
	    return (*rootp);		/* key found */
	rootp = (r < 0) ?
	    &(*rootp)->llink :		/* T3: follow left branch */
	    &(*rootp)->rlink;		/* T4: follow right branch */
    }
    return (node *)0;
}

