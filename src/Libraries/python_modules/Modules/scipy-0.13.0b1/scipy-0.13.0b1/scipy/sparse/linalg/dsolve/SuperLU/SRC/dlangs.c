/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
 * File name:	dlangs.c
 * History:     Modified from lapack routine DLANGE
 */
#include <math.h>
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose   
 *   =======   
 *
 *   DLANGS returns the value of the one norm, or the Frobenius norm, or 
 *   the infinity norm, or the element of largest absolute value of a 
 *   real matrix A.   
 *
 *   Description   
 *   ===========   
 *
 *   DLANGE returns the value   
 *
 *      DLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'   
 *               (   
 *               ( norm1(A),         NORM = '1', 'O' or 'o'   
 *               (   
 *               ( normI(A),         NORM = 'I' or 'i'   
 *               (   
 *               ( normF(A),         NORM = 'F', 'f', 'E' or 'e'   
 *
 *   where  norm1  denotes the  one norm of a matrix (maximum column sum), 
 *   normI  denotes the  infinity norm  of a matrix  (maximum row sum) and 
 *   normF  denotes the  Frobenius norm of a matrix (square root of sum of 
 *   squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm.   
 *
 *   Arguments   
 *   =========   
 *
 *   NORM    (input) CHARACTER*1   
 *           Specifies the value to be returned in DLANGE as described above.   
 *   A       (input) SuperMatrix*
 *           The M by N sparse matrix A. 
 *
 *  =====================================================================
 * </pre>
 */

double dlangs(char *norm, SuperMatrix *A)
{
    
    /* Local variables */
    NCformat *Astore;
    double   *Aval;
    int      i, j, irow;
    double   value, sum;
    double   *rwork;

    Astore = A->Store;
    Aval   = Astore->nzval;
    
    if ( SUPERLU_MIN(A->nrow, A->ncol) == 0) {
	value = 0.;
	
    } else if (lsame_(norm, "M")) {
	/* Find max(abs(A(i,j))). */
	value = 0.;
	for (j = 0; j < A->ncol; ++j)
	    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++)
		value = SUPERLU_MAX( value, fabs( Aval[i]) );
	
    } else if (lsame_(norm, "O") || *(unsigned char *)norm == '1') {
	/* Find norm1(A). */
	value = 0.;
	for (j = 0; j < A->ncol; ++j) {
	    sum = 0.;
	    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) 
		sum += fabs(Aval[i]);
	    value = SUPERLU_MAX(value,sum);
	}
	
    } else if (lsame_(norm, "I")) {
	/* Find normI(A). */
	if ( !(rwork = (double *) SUPERLU_MALLOC(A->nrow * sizeof(double))) )
	    ABORT("SUPERLU_MALLOC fails for rwork.");
	for (i = 0; i < A->nrow; ++i) rwork[i] = 0.;
	for (j = 0; j < A->ncol; ++j)
	    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; i++) {
		irow = Astore->rowind[i];
		rwork[irow] += fabs(Aval[i]);
	    }
	value = 0.;
	for (i = 0; i < A->nrow; ++i)
	    value = SUPERLU_MAX(value, rwork[i]);
	
	SUPERLU_FREE (rwork);
	
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {
	/* Find normF(A). */
	ABORT("Not implemented.");
    } else
	ABORT("Illegal norm specified.");

    return (value);

} /* dlangs */

