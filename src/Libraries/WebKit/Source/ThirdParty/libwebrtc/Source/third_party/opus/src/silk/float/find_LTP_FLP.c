/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
#include "tuning_parameters.h"

void silk_find_LTP_FLP(
    silk_float                      XX[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* O    Weight for LTP quantization         */
    silk_float                      xX[ MAX_NB_SUBFR * LTP_ORDER ],     /* O    Weight for LTP quantization                 */
    const silk_float                r_ptr[],                            /* I    LPC residual                                */
    const opus_int                  lag[ MAX_NB_SUBFR ],                /* I    LTP lags                                    */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr                            /* I    number of subframes                         */
)
{
    opus_int   k;
    silk_float *xX_ptr, *XX_ptr;
    const silk_float *lag_ptr;
    silk_float xx, temp;

    xX_ptr = xX;
    XX_ptr = XX;
    for( k = 0; k < nb_subfr; k++ ) {
        lag_ptr = r_ptr - ( lag[ k ] + LTP_ORDER / 2 );
        silk_corrMatrix_FLP( lag_ptr, subfr_length, LTP_ORDER, XX_ptr );
        silk_corrVector_FLP( lag_ptr, r_ptr, subfr_length, LTP_ORDER, xX_ptr );
        xx = ( silk_float )silk_energy_FLP( r_ptr, subfr_length + LTP_ORDER );
        temp = 1.0f / silk_max( xx, LTP_CORR_INV_MAX * 0.5f * ( XX_ptr[ 0 ] + XX_ptr[ 24 ] ) + 1.0f );
        silk_scale_vector_FLP( XX_ptr, temp, LTP_ORDER * LTP_ORDER );
        silk_scale_vector_FLP( xX_ptr, temp, LTP_ORDER );

        r_ptr  += subfr_length;
        XX_ptr += LTP_ORDER * LTP_ORDER;
        xX_ptr += LTP_ORDER;
    }
}
