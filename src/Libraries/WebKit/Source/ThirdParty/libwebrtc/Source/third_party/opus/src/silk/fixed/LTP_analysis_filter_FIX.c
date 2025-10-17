/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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

void silk_LTP_analysis_filter_FIX(
    opus_int16                      *LTP_res,                               /* O    LTP residual signal of length MAX_NB_SUBFR * ( pre_length + subfr_length )  */
    const opus_int16                *x,                                     /* I    Pointer to input signal with at least max( pitchL ) preceding samples       */
    const opus_int16                LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],/* I    LTP_ORDER LTP coefficients for each MAX_NB_SUBFR subframe                   */
    const opus_int                  pitchL[ MAX_NB_SUBFR ],                 /* I    Pitch lag, one for each subframe                                            */
    const opus_int32                invGains_Q16[ MAX_NB_SUBFR ],           /* I    Inverse quantization gains, one for each subframe                           */
    const opus_int                  subfr_length,                           /* I    Length of each subframe                                                     */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    const opus_int                  pre_length                              /* I    Length of the preceding samples starting at &x[0] for each subframe         */
)
{
    const opus_int16 *x_ptr, *x_lag_ptr;
    opus_int16   Btmp_Q14[ LTP_ORDER ];
    opus_int16   *LTP_res_ptr;
    opus_int     k, i;
    opus_int32   LTP_est;

    x_ptr = x;
    LTP_res_ptr = LTP_res;
    for( k = 0; k < nb_subfr; k++ ) {

        x_lag_ptr = x_ptr - pitchL[ k ];

        Btmp_Q14[ 0 ] = LTPCoef_Q14[ k * LTP_ORDER ];
        Btmp_Q14[ 1 ] = LTPCoef_Q14[ k * LTP_ORDER + 1 ];
        Btmp_Q14[ 2 ] = LTPCoef_Q14[ k * LTP_ORDER + 2 ];
        Btmp_Q14[ 3 ] = LTPCoef_Q14[ k * LTP_ORDER + 3 ];
        Btmp_Q14[ 4 ] = LTPCoef_Q14[ k * LTP_ORDER + 4 ];

        /* LTP analysis FIR filter */
        for( i = 0; i < subfr_length + pre_length; i++ ) {
            LTP_res_ptr[ i ] = x_ptr[ i ];

            /* Long-term prediction */
            LTP_est = silk_SMULBB( x_lag_ptr[ LTP_ORDER / 2 ], Btmp_Q14[ 0 ] );
            LTP_est = silk_SMLABB_ovflw( LTP_est, x_lag_ptr[ 1 ], Btmp_Q14[ 1 ] );
            LTP_est = silk_SMLABB_ovflw( LTP_est, x_lag_ptr[ 0 ], Btmp_Q14[ 2 ] );
            LTP_est = silk_SMLABB_ovflw( LTP_est, x_lag_ptr[ -1 ], Btmp_Q14[ 3 ] );
            LTP_est = silk_SMLABB_ovflw( LTP_est, x_lag_ptr[ -2 ], Btmp_Q14[ 4 ] );

            LTP_est = silk_RSHIFT_ROUND( LTP_est, 14 ); /* round and -> Q0*/

            /* Subtract long-term prediction */
            LTP_res_ptr[ i ] = (opus_int16)silk_SAT16( (opus_int32)x_ptr[ i ] - LTP_est );

            /* Scale residual */
            LTP_res_ptr[ i ] = silk_SMULWB( invGains_Q16[ k ], LTP_res_ptr[ i ] );

            x_lag_ptr++;
        }

        /* Update pointers */
        LTP_res_ptr += subfr_length + pre_length;
        x_ptr       += subfr_length;
    }
}

