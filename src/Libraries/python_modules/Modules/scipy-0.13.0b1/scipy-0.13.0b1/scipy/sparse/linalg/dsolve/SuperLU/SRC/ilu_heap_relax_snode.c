/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_heap_relax_snode() - Identify the initial relaxed supernodes,
 *    assuming that the matrix has been reordered according to the postorder
 *    of the etree.
 * </pre>
 */

void
ilu_heap_relax_snode (
	     const     int n,
	     int       *et,	      /* column elimination tree */
	     const int relax_columns, /* max no of columns allowed in a
					 relaxed snode */
	     int       *descendants,  /* no of descendants of each node
					 in the etree */
	     int       *relax_end,    /* last column in a supernode
				       * if j-th column starts a relaxed
				       * supernode, relax_end[j] represents
				       * the last column of this supernode */
	     int       *relax_fsupc   /* first column in a supernode
				       * relax_fsupc[j] represents the first
				       * column of j-th supernode */
	     )
{
    register int i, j, k, l, f, parent;
    register int snode_start;	/* beginning of a snode */
    int *et_save, *post, *inv_post, *iwork;
    int nsuper_et = 0, nsuper_et_post = 0;

    /* The etree may not be postordered, but is heap ordered. */

    iwork = (int*) intMalloc(3*n+2);
    if ( !iwork ) ABORT("SUPERLU_MALLOC fails for iwork[]");
    inv_post = iwork + n+1;
    et_save = inv_post + n+1;

    /* Post order etree */
    post = (int *) TreePostorder(n, et);
    for (i = 0; i < n+1; ++i) inv_post[post[i]] = i;

    /* Renumber etree in postorder */
    for (i = 0; i < n; ++i) {
	iwork[post[i]] = post[et[i]];
	et_save[i] = et[i]; /* Save the original etree */
    }
    for (i = 0; i < n; ++i) et[i] = iwork[i];

    /* Compute the number of descendants of each node in the etree */
    ifill (relax_end, n, EMPTY);
    ifill (relax_fsupc, n, EMPTY);
    for (j = 0; j < n; j++) descendants[j] = 0;
    for (j = 0; j < n; j++) {
	parent = et[j];
	if ( parent != n )  /* not the dummy root */
	    descendants[parent] += descendants[j] + 1;
    }

    /* Identify the relaxed supernodes by postorder traversal of the etree. */
    for ( f = j = 0; j < n; ) {
	parent = et[j];
	snode_start = j;
	while ( parent != n && descendants[parent] < relax_columns ) {
	    j = parent;
	    parent = et[j];
	}
	/* Found a supernode in postordered etree; j is the last column. */
	++nsuper_et_post;
	k = n;
	for (i = snode_start; i <= j; ++i)
	    k = SUPERLU_MIN(k, inv_post[i]);
	l = inv_post[j];
	if ( (l - k) == (j - snode_start) ) {
	    /* It's also a supernode in the original etree */
	    relax_end[k] = l;		/* Last column is recorded */
	    relax_fsupc[f++] = k;
	    ++nsuper_et;
	} else {
	    for (i = snode_start; i <= j; ++i) {
		l = inv_post[i];
		if ( descendants[i] == 0 ) {
		    relax_end[l] = l;
		    relax_fsupc[f++] = l;
		    ++nsuper_et;
		}
	    }
	}
	j++;
	/* Search for a new leaf */
	while ( descendants[j] != 0 && j < n ) j++;
    }

#if ( PRNTlevel>=1 )
    printf(".. heap_snode_relax:\n"
	   "\tNo of relaxed snodes in postordered etree:\t%d\n"
	   "\tNo of relaxed snodes in original etree:\t%d\n",
	   nsuper_et_post, nsuper_et);
#endif

    /* Recover the original etree */
    for (i = 0; i < n; ++i) et[i] = et_save[i];

    SUPERLU_FREE(post);
    SUPERLU_FREE(iwork);
}
