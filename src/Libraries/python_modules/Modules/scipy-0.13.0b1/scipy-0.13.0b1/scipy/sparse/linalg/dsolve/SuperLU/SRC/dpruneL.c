/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
 *   Prunes the L-structure of supernodes whose L-structure
 *   contains the current pivot row "pivrow"
 * </pre>
 */

void
dpruneL(
       const int  jcol,	     /* in */
       const int  *perm_r,   /* in */
       const int  pivrow,    /* in */
       const int  nseg,	     /* in */
       const int  *segrep,   /* in */
       const int  *repfnz,   /* in */
       int        *xprune,   /* out */
       GlobalLU_t *Glu       /* modified - global LU data structures */
       )
{

    double     utemp;
    int        jsupno, irep, irep1, kmin, kmax, krow, movnum;
    int        i, ktemp, minloc, maxloc;
    int        do_prune; /* logical variable */
    int        *xsup, *supno;
    int        *lsub, *xlsub;
    double     *lusup;
    int        *xlusup;

    xsup       = Glu->xsup;
    supno      = Glu->supno;
    lsub       = Glu->lsub;
    xlsub      = Glu->xlsub;
    lusup      = Glu->lusup;
    xlusup     = Glu->xlusup;
    
    /*
     * For each supernode-rep irep in U[*,j]
     */
    jsupno = supno[jcol];
    for (i = 0; i < nseg; i++) {

	irep = segrep[i];
	irep1 = irep + 1;
	do_prune = FALSE;

	/* Don't prune with a zero U-segment */
 	if ( repfnz[irep] == EMPTY )
		continue;

     	/* If a snode overlaps with the next panel, then the U-segment 
   	 * is fragmented into two parts -- irep and irep1. We should let
	 * pruning occur at the rep-column in irep1's snode. 
	 */
	if ( supno[irep] == supno[irep1] ) 	/* Don't prune */
		continue;

	/*
	 * If it has not been pruned & it has a nonz in row L[pivrow,i]
	 */
	if ( supno[irep] != jsupno ) {
	    if ( xprune[irep] >= xlsub[irep1] ) {
		kmin = xlsub[irep];
		kmax = xlsub[irep1] - 1;
		for (krow = kmin; krow <= kmax; krow++) 
		    if ( lsub[krow] == pivrow ) {
			do_prune = TRUE;
			break;
		    }
	    }
	    
    	    if ( do_prune ) {

	     	/* Do a quicksort-type partition
	     	 * movnum=TRUE means that the num values have to be exchanged.
	     	 */
	        movnum = FALSE;
	        if ( irep == xsup[supno[irep]] ) /* Snode of size 1 */
			movnum = TRUE;

	        while ( kmin <= kmax ) {

	    	    if ( perm_r[lsub[kmax]] == EMPTY ) 
			kmax--;
		    else if ( perm_r[lsub[kmin]] != EMPTY )
			kmin++;
		    else { /* kmin below pivrow (not yet pivoted), and kmax
                            * above pivrow: interchange the two subscripts
			    */
		        ktemp = lsub[kmin];
		        lsub[kmin] = lsub[kmax];
		        lsub[kmax] = ktemp;

			/* If the supernode has only one column, then we
 			 * only keep one set of subscripts. For any subscript 
			 * interchange performed, similar interchange must be 
			 * done on the numerical values.
 			 */
		        if ( movnum ) {
		    	    minloc = xlusup[irep] + (kmin - xlsub[irep]);
		    	    maxloc = xlusup[irep] + (kmax - xlsub[irep]);
			    utemp = lusup[minloc];
		  	    lusup[minloc] = lusup[maxloc];
			    lusup[maxloc] = utemp;
		        }

		        kmin++;
		        kmax--;

		    }

	        } /* while */

	        xprune[irep] = kmin;	/* Pruning */

#ifdef CHK_PRUNE
	printf("    After dpruneL(),using col %d:  xprune[%d] = %d\n", 
			jcol, irep, kmin);
#endif
	    } /* if do_prune */

	} /* if */

    } /* for each U-segment... */
}
