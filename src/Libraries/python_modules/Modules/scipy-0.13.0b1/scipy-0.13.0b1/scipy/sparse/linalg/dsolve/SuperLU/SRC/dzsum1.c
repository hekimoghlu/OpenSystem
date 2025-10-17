/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#include "slu_dcomplex.h"
#include "slu_Cnames.h"

/*! \brief

 <pre>
    Purpose   
    =======   

    DZSUM1 takes the sum of the absolute values of a complex   
    vector and returns a double precision result.   

    Based on DZASUM from the Level 1 BLAS.   
    The change is to use the 'genuine' absolute value.   

    Contributed by Nick Higham for use with ZLACON.   

    Arguments   
    =========   

    N       (input) INT   
            The number of elements in the vector CX.   

    CX      (input) COMPLEX*16 array, dimension (N)   
            The vector whose elements will be summed.   

    INCX    (input) INT   
            The spacing between successive values of CX.  INCX > 0.   

    ===================================================================== 
</pre>
*/  
double dzsum1_(int *n, doublecomplex *cx, int *incx)
{

    /* Builtin functions */
    double z_abs(doublecomplex *);
    
    /* Local variables */
    int i, nincx;
    double stemp;


#define CX(I) cx[(I)-1]

    stemp = 0.;
    if (*n <= 0) {
	return stemp;
    }
    if (*incx == 1) {
	goto L20;
    }

    /*     CODE FOR INCREMENT NOT EQUAL TO 1 */

    nincx = *n * *incx;
    for (i = 1; *incx < 0 ? i >= nincx : i <= nincx; i += *incx) {

	/*        NEXT LINE MODIFIED. */

	stemp += z_abs(&CX(i));
/* L10: */
    }
    
    return stemp;

    /*     CODE FOR INCREMENT EQUAL TO 1 */

L20:
    for (i = 1; i <= *n; ++i) {

	/*        NEXT LINE MODIFIED. */

	stemp += z_abs(&CX(i));
/* L30: */
    }
    
    return stemp;

    /*     End of DZSUM1 */

} /* dzsum1_ */

