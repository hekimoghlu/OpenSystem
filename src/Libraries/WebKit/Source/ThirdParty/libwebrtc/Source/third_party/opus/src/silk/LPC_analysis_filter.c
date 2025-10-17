/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#include "celt_lpc.h"

/*******************************************/
/* LPC analysis filter                     */
/* NB! State is kept internally and the    */
/* filter always starts with zero state    */
/* first d output samples are set to zero  */
/*******************************************/

/* OPT: Using celt_fir() for this function should be faster, but it may cause
   integer overflows in intermediate values (not final results), which the
   current implementation silences by casting to unsigned. Enabling
   this should be safe in pretty much all cases, even though it is not technically
   C89-compliant. */
#define USE_CELT_FIR 0

void silk_LPC_analysis_filter(
    opus_int16                  *out,               /* O    Output signal                                               */
    const opus_int16            *in,                /* I    Input signal                                                */
    const opus_int16            *B,                 /* I    MA prediction coefficients, Q12 [order]                     */
    const opus_int32            len,                /* I    Signal length                                               */
    const opus_int32            d,                  /* I    Filter order                                                */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    opus_int   j;
#if defined(FIXED_POINT) && USE_CELT_FIR
    opus_int16 num[SILK_MAX_ORDER_LPC];
#else
    int ix;
    opus_int32       out32_Q12, out32;
    const opus_int16 *in_ptr;
#endif

    celt_assert( d >= 6 );
    celt_assert( (d & 1) == 0 );
    celt_assert( d <= len );

#if defined(FIXED_POINT) && USE_CELT_FIR
    celt_assert( d <= SILK_MAX_ORDER_LPC );
    for ( j = 0; j < d; j++ ) {
        num[ j ] = -B[ j ];
    }
    celt_fir( in + d, num, out + d, len - d, d, arch );
    for ( j = 0; j < d; j++ ) {
        out[ j ] = 0;
    }
#else
    (void)arch;
    for( ix = d; ix < len; ix++ ) {
        in_ptr = &in[ ix - 1 ];

        out32_Q12 = silk_SMULBB( in_ptr[  0 ], B[ 0 ] );
        /* Allowing wrap around so that two wraps can cancel each other. The rare
           cases where the result wraps around can only be triggered by invalid streams*/
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -1 ], B[ 1 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -2 ], B[ 2 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -3 ], B[ 3 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -4 ], B[ 4 ] );
        out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -5 ], B[ 5 ] );
        for( j = 6; j < d; j += 2 ) {
            out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -j     ], B[ j     ] );
            out32_Q12 = silk_SMLABB_ovflw( out32_Q12, in_ptr[ -j - 1 ], B[ j + 1 ] );
        }

        /* Subtract prediction */
        out32_Q12 = silk_SUB32_ovflw( silk_LSHIFT( (opus_int32)in_ptr[ 1 ], 12 ), out32_Q12 );

        /* Scale to Q0 */
        out32 = silk_RSHIFT_ROUND( out32_Q12, 12 );

        /* Saturate output */
        out[ ix ] = (opus_int16)silk_SAT16( out32 );
    }

    /* Set first d output samples to zero */
    silk_memset( out, 0, d * sizeof( opus_int16 ) );
#endif
}
