/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
 *    relax_snode() - Identify the initial relaxed supernodes, assuming that 
 *    the matrix has been reordered according to the postorder of the etree.
 * </pre>
 */ 
void
relax_snode (
	     const     int n,
	     int       *et,           /* column elimination tree */
	     const int relax_columns, /* max no of columns allowed in a
					 relaxed snode */
	     int       *descendants,  /* no of descendants of each node
					 in the etree */
	     int       *relax_end     /* last column in a supernode */
	     )
{

    register int j, parent;
    register int snode_start;	/* beginning of a snode */
    
    ifill (relax_end, n, EMPTY);
    for (j = 0; j < n; j++) descendants[j] = 0;

    /* Compute the number of descendants of each node in the etree */
    for (j = 0; j < n; j++) {
	parent = et[j];
	if ( parent != n )  /* not the dummy root */
	    descendants[parent] += descendants[j] + 1;
    }

    /* Identify the relaxed supernodes by postorder traversal of the etree. */
    for (j = 0; j < n; ) { 
     	parent = et[j];
        snode_start = j;
 	while ( parent != n && descendants[parent] < relax_columns ) {
	    j = parent;
	    parent = et[j];
	}
	/* Found a supernode with j being the last column. */
	relax_end[snode_start] = j;		/* Last column is recorded */
	j++;
	/* Search for a new leaf */
	while ( descendants[j] != 0 && j < n ) j++;
    }

    /*printf("No of relaxed snodes: %d; relaxed columns: %d\n", 
		nsuper, no_relaxed_col); */
}
