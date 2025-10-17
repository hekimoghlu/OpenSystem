/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
#include "resampler_private.h"

/* Second order AR filter with single delay elements */
void silk_resampler_private_AR2(
    opus_int32                      S[],            /* I/O  State vector [ 2 ]          */
    opus_int32                      out_Q8[],       /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    const opus_int16                A_Q14[],        /* I    AR coefficients, Q14        */
    opus_int32                      len             /* I    Signal length               */
)
{
    opus_int32    k;
    opus_int32    out32;

    for( k = 0; k < len; k++ ) {
        out32       = silk_ADD_LSHIFT32( S[ 0 ], (opus_int32)in[ k ], 8 );
        out_Q8[ k ] = out32;
        out32       = silk_LSHIFT( out32, 2 );
        S[ 0 ]      = silk_SMLAWB( S[ 1 ], out32, A_Q14[ 0 ] );
        S[ 1 ]      = silk_SMULWB( out32, A_Q14[ 1 ] );
    }
}

