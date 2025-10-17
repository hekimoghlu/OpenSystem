/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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

/* Convert int32 coefficients to int16 coefs and make sure there's no wrap-around.
   This logic is reused in _celt_lpc(). Any bug fixes should also be applied there. */
void silk_LPC_fit(
    opus_int16                  *a_QOUT,            /* O    Output signal                                               */
    opus_int32                    *a_QIN,             /* I/O  Input signal                                                */
    const opus_int              QOUT,               /* I    Input Q domain                                              */
    const opus_int              QIN,                /* I    Input Q domain                                              */
    const opus_int              d                   /* I    Filter order                                                */
)
{
    opus_int    i, k, idx = 0;
    opus_int32    maxabs, absval, chirp_Q16;

    /* Limit the maximum absolute value of the prediction coefficients, so that they'll fit in int16 */
    for( i = 0; i < 10; i++ ) {
        /* Find maximum absolute value and its index */
        maxabs = 0;
        for( k = 0; k < d; k++ ) {
            absval = silk_abs( a_QIN[k] );
            if( absval > maxabs ) {
                maxabs = absval;
                idx    = k;
            }
        }
        maxabs = silk_RSHIFT_ROUND( maxabs, QIN - QOUT );

        if( maxabs > silk_int16_MAX ) {
            /* Reduce magnitude of prediction coefficients */
            maxabs = silk_min( maxabs, 163838 );  /* ( silk_int32_MAX >> 14 ) + silk_int16_MAX = 163838 */
            chirp_Q16 = SILK_FIX_CONST( 0.999, 16 ) - silk_DIV32( silk_LSHIFT( maxabs - silk_int16_MAX, 14 ),
                                        silk_RSHIFT32( silk_MUL( maxabs, idx + 1), 2 ) );
            silk_bwexpander_32( a_QIN, d, chirp_Q16 );
        } else {
            break;
        }
    }

    if( i == 10 ) {
        /* Reached the last iteration, clip the coefficients */
        for( k = 0; k < d; k++ ) {
            a_QOUT[ k ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( a_QIN[ k ], QIN - QOUT ) );
            a_QIN[ k ] = silk_LSHIFT( (opus_int32)a_QOUT[ k ], QIN - QOUT );
        }
    } else {
        for( k = 0; k < d; k++ ) {
            a_QOUT[ k ] = (opus_int16)silk_RSHIFT_ROUND( a_QIN[ k ], QIN - QOUT );
        }
    }
}
