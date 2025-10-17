/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#include "slu_zdefs.h"

int
zcopy_to_ucol(
	      int        jcol,	  /* in */
	      int        nseg,	  /* in */
	      int        *segrep,  /* in */
	      int        *repfnz,  /* in */
	      int        *perm_r,  /* in */
	      doublecomplex     *dense,   /* modified - reset to zero on return */
	      GlobalLU_t *Glu      /* modified */
	      )
{
/* 
 * Gather from SPA dense[*] to global ucol[*].
 */
    int ksub, krep, ksupno;
    int i, k, kfnz, segsze;
    int fsupc, isub, irow;
    int jsupno, nextu;
    int new_next, mem_error;
    int       *xsup, *supno;
    int       *lsub, *xlsub;
    doublecomplex    *ucol;
    int       *usub, *xusub;
    int       nzumax;
    doublecomplex zero = {0.0, 0.0};

    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    ucol    = Glu->ucol;
    usub    = Glu->usub;
    xusub   = Glu->xusub;
    nzumax  = Glu->nzumax;
    
    jsupno = supno[jcol];
    nextu  = xusub[jcol];
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) {
	krep = segrep[k--];
	ksupno = supno[krep];

	if ( ksupno != jsupno ) { /* Should go into ucol[] */
	    kfnz = repfnz[krep];
	    if ( kfnz != EMPTY ) {	/* Nonzero U-segment */

	    	fsupc = xsup[ksupno];
	        isub = xlsub[fsupc] + kfnz - fsupc;
	        segsze = krep - kfnz + 1;

		new_next = nextu + segsze;
		while ( new_next > nzumax ) {
		    if (mem_error = zLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu))
			return (mem_error);
		    ucol = Glu->ucol;
		    if (mem_error = zLUMemXpand(jcol, nextu, USUB, &nzumax, Glu))
			return (mem_error);
		    usub = Glu->usub;
		    lsub = Glu->lsub;
		}
		
		for (i = 0; i < segsze; i++) {
		    irow = lsub[isub];
		    usub[nextu] = perm_r[irow];
		    ucol[nextu] = dense[irow];
		    dense[irow] = zero;
		    nextu++;
		    isub++;
		} 

	    }

	}

    } /* for each segment... */

    xusub[jcol + 1] = nextu;      /* Close U[*,jcol] */
    return 0;
}
