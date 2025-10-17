/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "tuning_parameters.h"

void silk_find_LTP_FIX(
    opus_int32                      XXLTP_Q17[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Correlation matrix                                               */
    opus_int32                      xXLTP_Q17[ MAX_NB_SUBFR * LTP_ORDER ],  /* O    Correlation vector                                                          */
    const opus_int16                r_ptr[],                                /* I    Residual signal after LPC                                                   */
    const opus_int                  lag[ MAX_NB_SUBFR ],                    /* I    LTP lags                                                                    */
    const opus_int                  subfr_length,                           /* I    Subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    int                             arch                                    /* I    Run-time architecture                                                       */
)
{
    opus_int   i, k, extra_shifts;
    opus_int   xx_shifts, xX_shifts, XX_shifts;
    const opus_int16 *lag_ptr;
    opus_int32 *XXLTP_Q17_ptr, *xXLTP_Q17_ptr;
    opus_int32 xx, nrg, temp;

    xXLTP_Q17_ptr = xXLTP_Q17;
    XXLTP_Q17_ptr = XXLTP_Q17;
    for( k = 0; k < nb_subfr; k++ ) {
        lag_ptr = r_ptr - ( lag[ k ] + LTP_ORDER / 2 );

        silk_sum_sqr_shift( &xx, &xx_shifts, r_ptr, subfr_length + LTP_ORDER );                            /* xx in Q( -xx_shifts ) */
        silk_corrMatrix_FIX( lag_ptr, subfr_length, LTP_ORDER, XXLTP_Q17_ptr, &nrg, &XX_shifts, arch );    /* XXLTP_Q17_ptr and nrg in Q( -XX_shifts ) */
        extra_shifts = xx_shifts - XX_shifts;
        if( extra_shifts > 0 ) {
            /* Shift XX */
            xX_shifts = xx_shifts;
            for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
                XXLTP_Q17_ptr[ i ] = silk_RSHIFT32( XXLTP_Q17_ptr[ i ], extra_shifts );              /* Q( -xX_shifts ) */
            }
            nrg = silk_RSHIFT32( nrg, extra_shifts );                                                /* Q( -xX_shifts ) */
        } else if( extra_shifts < 0 ) {
            /* Shift xx */
            xX_shifts = XX_shifts;
            xx = silk_RSHIFT32( xx, -extra_shifts );                                                 /* Q( -xX_shifts ) */
        } else {
            xX_shifts = xx_shifts;
        }
        silk_corrVector_FIX( lag_ptr, r_ptr, subfr_length, LTP_ORDER, xXLTP_Q17_ptr, xX_shifts, arch );    /* xXLTP_Q17_ptr in Q( -xX_shifts ) */

        /* At this point all correlations are in Q(-xX_shifts) */
        temp = silk_SMLAWB( 1, nrg, SILK_FIX_CONST( LTP_CORR_INV_MAX, 16 ) );
        temp = silk_max( temp, xx );
TIC(div)
#if 0
        for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
            XXLTP_Q17_ptr[ i ] = silk_DIV32_varQ( XXLTP_Q17_ptr[ i ], temp, 17 );
        }
        for( i = 0; i < LTP_ORDER; i++ ) {
            xXLTP_Q17_ptr[ i ] = silk_DIV32_varQ( xXLTP_Q17_ptr[ i ], temp, 17 );
        }
#else
        for( i = 0; i < LTP_ORDER * LTP_ORDER; i++ ) {
            XXLTP_Q17_ptr[ i ] = (opus_int32)( silk_LSHIFT64( (opus_int64)XXLTP_Q17_ptr[ i ], 17 ) / temp );
        }
        for( i = 0; i < LTP_ORDER; i++ ) {
            xXLTP_Q17_ptr[ i ] = (opus_int32)( silk_LSHIFT64( (opus_int64)xXLTP_Q17_ptr[ i ], 17 ) / temp );
        }
#endif
TOC(div)
        r_ptr         += subfr_length;
        XXLTP_Q17_ptr += LTP_ORDER * LTP_ORDER;
        xXLTP_Q17_ptr += LTP_ORDER;
    }
}
