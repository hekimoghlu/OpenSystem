/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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


/*! \brief Performs numeric block updates within the relaxed snode. 
 */
int
dsnode_bmod (
	    const int  jcol,	  /* in */
	    const int  jsupno,    /* in */
	    const int  fsupc,     /* in */
	    double     *dense,    /* in */
	    double     *tempv,    /* working array */
	    GlobalLU_t *Glu,      /* modified */
	    SuperLUStat_t *stat   /* output */
	    )
{
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
	 ftcs2 = _cptofcd("N", strlen("N")),
	 ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int            incx = 1, incy = 1;
    double         alpha = -1.0, beta = 1.0;
#endif

    int            luptr, nsupc, nsupr, nrow;
    int            isub, irow, i, iptr; 
    register int   ufirst, nextlu;
    int            *lsub, *xlsub;
    double         *lusup;
    int            *xlusup;
    flops_t *ops = stat->ops;

    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = Glu->lusup;
    xlusup  = Glu->xlusup;

    nextlu = xlusup[jcol];
    
    /*
     *	Process the supernodal portion of L\U[*,j]
     */
    for (isub = xlsub[fsupc]; isub < xlsub[fsupc+1]; isub++) {
  	irow = lsub[isub];
	lusup[nextlu] = dense[irow];
	dense[irow] = 0;
	++nextlu;
    }

    xlusup[jcol + 1] = nextlu;	/* Initialize xlusup for next column */
    
    if ( fsupc < jcol ) {

	luptr = xlusup[fsupc];
	nsupr = xlsub[fsupc+1] - xlsub[fsupc];
	nsupc = jcol - fsupc;	/* Excluding jcol */
	ufirst = xlusup[jcol];	/* Points to the beginning of column
				   jcol in supernode L\U(jsupno). */
	nrow = nsupr - nsupc;

	ops[TRSV] += nsupc * (nsupc - 1);
	ops[GEMV] += 2 * nrow * nsupc;

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
	STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &lusup[luptr], &nsupr, 
	      &lusup[ufirst], &incx );
	SGEMV( ftcs2, &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
		&lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#else
	dtrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr, 
	      &lusup[ufirst], &incx );
	dgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, 
		&lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy );
#endif
#else
	dlsolve ( nsupr, nsupc, &lusup[luptr], &lusup[ufirst] );
	dmatvec ( nsupr, nrow, nsupc, &lusup[luptr+nsupc], 
			&lusup[ufirst], &tempv[0] );

        /* Scatter tempv[*] into lusup[*] */
	iptr = ufirst + nsupc;
	for (i = 0; i < nrow; i++) {
	    lusup[iptr++] -= tempv[i];
	    tempv[i] = 0.0;
	}
#endif

    }

    return 0;
}
