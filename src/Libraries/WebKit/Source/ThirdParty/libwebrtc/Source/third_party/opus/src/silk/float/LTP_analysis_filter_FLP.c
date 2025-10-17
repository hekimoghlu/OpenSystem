/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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

#include "main_FLP.h"

void silk_LTP_analysis_filter_FLP(
    silk_float                      *LTP_res,                           /* O    LTP res MAX_NB_SUBFR*(pre_lgth+subfr_lngth) */
    const silk_float                *x,                                 /* I    Input signal, with preceding samples        */
    const silk_float                B[ LTP_ORDER * MAX_NB_SUBFR ],      /* I    LTP coefficients for each subframe          */
    const opus_int                  pitchL[   MAX_NB_SUBFR ],           /* I    Pitch lags                                  */
    const silk_float                invGains[ MAX_NB_SUBFR ],           /* I    Inverse quantization gains                  */
    const opus_int                  subfr_length,                       /* I    Length of each subframe                     */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    const opus_int                  pre_length                          /* I    Preceding samples for each subframe         */
)
{
    const silk_float *x_ptr, *x_lag_ptr;
    silk_float   Btmp[ LTP_ORDER ];
    silk_float   *LTP_res_ptr;
    silk_float   inv_gain;
    opus_int     k, i, j;

    x_ptr = x;
    LTP_res_ptr = LTP_res;
    for( k = 0; k < nb_subfr; k++ ) {
        x_lag_ptr = x_ptr - pitchL[ k ];
        inv_gain = invGains[ k ];
        for( i = 0; i < LTP_ORDER; i++ ) {
            Btmp[ i ] = B[ k * LTP_ORDER + i ];
        }

        /* LTP analysis FIR filter */
        for( i = 0; i < subfr_length + pre_length; i++ ) {
            LTP_res_ptr[ i ] = x_ptr[ i ];
            /* Subtract long-term prediction */
            for( j = 0; j < LTP_ORDER; j++ ) {
                LTP_res_ptr[ i ] -= Btmp[ j ] * x_lag_ptr[ LTP_ORDER / 2 - j ];
            }
            LTP_res_ptr[ i ] *= inv_gain;
            x_lag_ptr++;
        }

        /* Update pointers */
        LTP_res_ptr += subfr_length + pre_length;
        x_ptr       += subfr_length;
    }
}
