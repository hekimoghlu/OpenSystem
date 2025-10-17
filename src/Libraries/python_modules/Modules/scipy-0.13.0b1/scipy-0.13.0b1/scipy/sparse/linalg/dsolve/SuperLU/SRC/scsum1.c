/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#include "slu_scomplex.h"
#include "slu_Cnames.h"

/*! \brief

<pre>
    Purpose   
    =======   

    SCSUM1 takes the sum of the absolute values of a complex   
    vector and returns a single precision result.   

    Based on SCASUM from the Level 1 BLAS.   
    The change is to use the 'genuine' absolute value.   

    Contributed by Nick Higham for use with CLACON.   

    Arguments   
    =========   

    N       (input) INT
            The number of elements in the vector CX.   

    CX      (input) COMPLEX array, dimension (N)   
            The vector whose elements will be summed.   

    INCX    (input) INT
            The spacing between successive values of CX.  INCX > 0.   

    ===================================================================== 
</pre>
*/
double scsum1_(int *n, complex *cx, int *incx)
{
    /* System generated locals */
    int i__1, i__2;
    float ret_val;
    /* Builtin functions */
    double c_abs(complex *);
    /* Local variables */
    static int i, nincx;
    static float stemp;


#define CX(I) cx[(I)-1]


    ret_val = 0.f;
    stemp = 0.f;
    if (*n <= 0) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*     CODE FOR INCREMENT NOT EQUAL TO 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i = 1; *incx < 0 ? i >= nincx : i <= nincx; i += *incx) {

/*        NEXT LINE MODIFIED. */

	stemp += c_abs(&CX(i));
/* L10: */
    }
    ret_val = stemp;
    return ret_val;

/*     CODE FOR INCREMENT EQUAL TO 1 */

L20:
    i__2 = *n;
    for (i = 1; i <= *n; ++i) {

/*        NEXT LINE MODIFIED. */

	stemp += c_abs(&CX(i));
/* L30: */
    }
    ret_val = stemp;
    return ret_val;

/*     End of SCSUM1 */

} /* scsum1_ */

