/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

/* Slower than schur(), but more accurate.                              */
/* Uses SMULL(), available on armv4                                     */
opus_int32 silk_schur64(                            /* O    returns residual energy                                     */
    opus_int32                  rc_Q16[],           /* O    Reflection coefficients [order] Q16                         */
    const opus_int32            c[],                /* I    Correlations [order+1]                                      */
    opus_int32                  order               /* I    Prediction order                                            */
)
{
    opus_int   k, n;
    opus_int32 C[ SILK_MAX_ORDER_LPC + 1 ][ 2 ];
    opus_int32 Ctmp1_Q30, Ctmp2_Q30, rc_tmp_Q31;

    celt_assert( order >= 0 && order <= SILK_MAX_ORDER_LPC );

    /* Check for invalid input */
    if( c[ 0 ] <= 0 ) {
        silk_memset( rc_Q16, 0, order * sizeof( opus_int32 ) );
        return 0;
    }

    k = 0;
    do {
        C[ k ][ 0 ] = C[ k ][ 1 ] = c[ k ];
    } while( ++k <= order );

    for( k = 0; k < order; k++ ) {
        /* Check that we won't be getting an unstable rc, otherwise stop here. */
        if (silk_abs_int32(C[ k + 1 ][ 0 ]) >= C[ 0 ][ 1 ]) {
           if ( C[ k + 1 ][ 0 ] > 0 ) {
              rc_Q16[ k ] = -SILK_FIX_CONST( .99f, 16 );
           } else {
              rc_Q16[ k ] = SILK_FIX_CONST( .99f, 16 );
           }
           k++;
           break;
        }

        /* Get reflection coefficient: divide two Q30 values and get result in Q31 */
        rc_tmp_Q31 = silk_DIV32_varQ( -C[ k + 1 ][ 0 ], C[ 0 ][ 1 ], 31 );

        /* Save the output */
        rc_Q16[ k ] = silk_RSHIFT_ROUND( rc_tmp_Q31, 15 );

        /* Update correlations */
        for( n = 0; n < order - k; n++ ) {
            Ctmp1_Q30 = C[ n + k + 1 ][ 0 ];
            Ctmp2_Q30 = C[ n ][ 1 ];

            /* Multiply and add the highest int32 */
            C[ n + k + 1 ][ 0 ] = Ctmp1_Q30 + silk_SMMUL( silk_LSHIFT( Ctmp2_Q30, 1 ), rc_tmp_Q31 );
            C[ n ][ 1 ]         = Ctmp2_Q30 + silk_SMMUL( silk_LSHIFT( Ctmp1_Q30, 1 ), rc_tmp_Q31 );
        }
    }

    for(; k < order; k++ ) {
       rc_Q16[ k ] = 0;
    }

    return silk_max_32( 1, C[ 0 ][ 1 ] );
}
