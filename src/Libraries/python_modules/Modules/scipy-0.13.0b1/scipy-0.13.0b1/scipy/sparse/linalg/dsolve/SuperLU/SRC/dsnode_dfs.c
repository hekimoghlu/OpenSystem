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
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    dsnode_dfs() - Determine the union of the row structures of those 
 *    columns within the relaxed snode.
 *    Note: The relaxed snodes are leaves of the supernodal etree, therefore, 
 *    the portion outside the rectangular supernode must be zero.
 *
 * Return value
 * ============
 *     0   success;
 *    >0   number of bytes allocated when run out of memory.
 * </pre>
 */

int
dsnode_dfs (
	   const int  jcol,	    /* in - start of the supernode */
	   const int  kcol, 	    /* in - end of the supernode */
	   const int  *asub,        /* in */
	   const int  *xa_begin,    /* in */
	   const int  *xa_end,      /* in */
	   int        *xprune,      /* out */
	   int        *marker,      /* modified */
	   GlobalLU_t *Glu          /* modified */
	   )
{

    register int i, k, ifrom, ito, nextl, new_next;
    int          nsuper, krow, kmark, mem_error;
    int          *xsup, *supno;
    int          *lsub, *xlsub;
    int          nzlmax;
    
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    nzlmax  = Glu->nzlmax;

    nsuper = ++supno[jcol];	/* Next available supernode number */
    nextl = xlsub[jcol];

    for (i = jcol; i <= kcol; i++) {
	/* For each nonzero in A[*,i] */
	for (k = xa_begin[i]; k < xa_end[i]; k++) {	
	    krow = asub[k];
	    kmark = marker[krow];
	    if ( kmark != kcol ) { /* First time visit krow */
		marker[krow] = kcol;
		lsub[nextl++] = krow;
		if ( nextl >= nzlmax ) {
		    if ( mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
			return (mem_error);
		    lsub = Glu->lsub;
		}
	    }
    	}
	supno[i] = nsuper;
    }

    /* Supernode > 1, then make a copy of the subscripts for pruning */
    if ( jcol < kcol ) {
	new_next = nextl + (nextl - xlsub[jcol]);
	while ( new_next > nzlmax ) {
	    if ( mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
		return (mem_error);
	    lsub = Glu->lsub;
	}
	ito = nextl;
	for (ifrom = xlsub[jcol]; ifrom < nextl; )
	    lsub[ito++] = lsub[ifrom++];	
        for (i = jcol+1; i <= kcol; i++) xlsub[i] = nextl;
	nextl = ito;
    }

    xsup[nsuper+1] = kcol + 1;
    supno[kcol+1]  = nsuper;
    xprune[kcol]   = nextl;
    xlsub[kcol+1]  = nextl;

    return 0;
}

