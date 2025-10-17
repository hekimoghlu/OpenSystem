/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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

#include "main_FIX.h"

#if defined(MIPSr1_ASM)
#include "mips/warped_autocorrelation_FIX_mipsr1.h"
#endif


/* Autocorrelations for a warped frequency axis */
#ifndef OVERRIDE_silk_warped_autocorrelation_FIX_c
void silk_warped_autocorrelation_FIX_c(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
)
{
    opus_int   n, i, lsh;
    opus_int32 tmp1_QS, tmp2_QS;
    opus_int32 state_QS[ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    opus_int64 corr_QC[  MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };

    /* Order must be even */
    celt_assert( ( order & 1 ) == 0 );
    silk_assert( 2 * QS - QC >= 0 );

    /* Loop over samples */
    for( n = 0; n < length; n++ ) {
        tmp1_QS = silk_LSHIFT32( (opus_int32)input[ n ], QS );
        /* Loop over allpass sections */
        for( i = 0; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2_QS = silk_SMLAWB( state_QS[ i ], state_QS[ i + 1 ] - tmp1_QS, warping_Q16 );
            state_QS[ i ]  = tmp1_QS;
            corr_QC[  i ] += silk_RSHIFT64( silk_SMULL( tmp1_QS, state_QS[ 0 ] ), 2 * QS - QC );
            /* Output of allpass section */
            tmp1_QS = silk_SMLAWB( state_QS[ i + 1 ], state_QS[ i + 2 ] - tmp2_QS, warping_Q16 );
            state_QS[ i + 1 ]  = tmp2_QS;
            corr_QC[  i + 1 ] += silk_RSHIFT64( silk_SMULL( tmp2_QS, state_QS[ 0 ] ), 2 * QS - QC );
        }
        state_QS[ order ] = tmp1_QS;
        corr_QC[  order ] += silk_RSHIFT64( silk_SMULL( tmp1_QS, state_QS[ 0 ] ), 2 * QS - QC );
    }

    lsh = silk_CLZ64( corr_QC[ 0 ] ) - 35;
    lsh = silk_LIMIT( lsh, -12 - QC, 30 - QC );
    *scale = -( QC + lsh );
    silk_assert( *scale >= -30 && *scale <= 12 );
    if( lsh >= 0 ) {
        for( i = 0; i < order + 1; i++ ) {
            corr[ i ] = (opus_int32)silk_CHECK_FIT32( silk_LSHIFT64( corr_QC[ i ], lsh ) );
        }
    } else {
        for( i = 0; i < order + 1; i++ ) {
            corr[ i ] = (opus_int32)silk_CHECK_FIT32( silk_RSHIFT64( corr_QC[ i ], -lsh ) );
        }
    }
    silk_assert( corr_QC[ 0 ] >= 0 ); /* If breaking, decrease QC*/
}
#endif /* OVERRIDE_silk_warped_autocorrelation_FIX_c */
