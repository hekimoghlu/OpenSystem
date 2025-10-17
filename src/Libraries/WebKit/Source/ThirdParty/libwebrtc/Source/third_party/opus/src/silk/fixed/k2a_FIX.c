/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FIX.h"

/* Step up function, converts reflection coefficients to prediction coefficients */
void silk_k2a(
    opus_int32                  *A_Q24,             /* O    Prediction coefficients [order] Q24                         */
    const opus_int16            *rc_Q15,            /* I    Reflection coefficients [order] Q15                         */
    const opus_int32            order               /* I    Prediction order                                            */
)
{
    opus_int   k, n;
    opus_int32 rc, tmp1, tmp2;

    for( k = 0; k < order; k++ ) {
        rc = rc_Q15[ k ];
        for( n = 0; n < (k + 1) >> 1; n++ ) {
            tmp1 = A_Q24[ n ];
            tmp2 = A_Q24[ k - n - 1 ];
            A_Q24[ n ]         = silk_SMLAWB( tmp1, silk_LSHIFT( tmp2, 1 ), rc );
            A_Q24[ k - n - 1 ] = silk_SMLAWB( tmp2, silk_LSHIFT( tmp1, 1 ), rc );
        }
        A_Q24[ k ] = -silk_LSHIFT( (opus_int32)rc_Q15[ k ], 9 );
    }
}
