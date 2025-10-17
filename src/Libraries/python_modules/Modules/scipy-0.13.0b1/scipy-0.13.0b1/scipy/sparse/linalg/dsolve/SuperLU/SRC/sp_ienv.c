/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
 * File name:		sp_ienv.c
 * History:             Modified from lapack routine ILAENV
 */
#include "slu_Cnames.h"

/*! \brief

 <pre>
    Purpose   
    =======   

    sp_ienv() is inquired to choose machine-dependent parameters for the
    local environment. See ISPEC for a description of the parameters.   

    This version provides a set of parameters which should give good,   
    but not optimal, performance on many of the currently available   
    computers.  Users are encouraged to modify this subroutine to set   
    the tuning parameters for their particular machine using the option   
    and problem size information in the arguments.   

    Arguments   
    =========   

    ISPEC   (input) int
            Specifies the parameter to be returned as the value of SP_IENV.   
            = 1: the panel size w; a panel consists of w consecutive
	         columns of matrix A in the process of Gaussian elimination.
		 The best value depends on machine's cache characters.
            = 2: the relaxation parameter relax; if the number of
	         nodes (columns) in a subtree of the elimination tree is less
		 than relax, this subtree is considered as one supernode,
		 regardless of their row structures.
            = 3: the maximum size for a supernode in complete LU;
	    = 4: the minimum row dimension for 2-D blocking to be used;
	    = 5: the minimum column dimension for 2-D blocking to be used;
	    = 6: the estimated fills factor for L and U, compared with A;
	    = 7: the maximum size for a supernode in ILU.
	    
   (SP_IENV) (output) int
            >= 0: the value of the parameter specified by ISPEC   
            < 0:  if SP_IENV = -k, the k-th argument had an illegal value. 
  
    ===================================================================== 
</pre>
*/
int
sp_ienv(int ispec)
{
    int i;

    switch (ispec) {
	case 1: return (12);
	case 2: return (6);
	case 3: return (100);
	case 4: return (200);
	case 5: return (60);
        case 6: return (20);
        case 7: return (10);
    }

    /* Invalid value for ISPEC */
    i = 1;
    xerbla_("sp_ienv", &i);
    return 0;

} /* sp_ienv_ */

