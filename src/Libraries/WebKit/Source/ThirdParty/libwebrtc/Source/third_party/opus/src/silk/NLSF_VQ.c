/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

#include "main.h"

/* Compute quantization errors for an LPC_order element input vector for a VQ codebook */
void silk_NLSF_VQ(
    opus_int32                  err_Q24[],                      /* O    Quantization errors [K]                     */
    const opus_int16            in_Q15[],                       /* I    Input vectors to be quantized [LPC_order]   */
    const opus_uint8            pCB_Q8[],                       /* I    Codebook vectors [K*LPC_order]              */
    const opus_int16            pWght_Q9[],                     /* I    Codebook weights [K*LPC_order]              */
    const opus_int              K,                              /* I    Number of codebook vectors                  */
    const opus_int              LPC_order                       /* I    Number of LPCs                              */
)
{
    opus_int         i, m;
    opus_int32       diff_Q15, diffw_Q24, sum_error_Q24, pred_Q24;
    const opus_int16 *w_Q9_ptr;
    const opus_uint8 *cb_Q8_ptr;

    celt_assert( ( LPC_order & 1 ) == 0 );

    /* Loop over codebook */
    cb_Q8_ptr = pCB_Q8;
    w_Q9_ptr = pWght_Q9;
    for( i = 0; i < K; i++ ) {
        sum_error_Q24 = 0;
        pred_Q24 = 0;
        for( m = LPC_order-2; m >= 0; m -= 2 ) {
            /* Compute weighted absolute predictive quantization error for index m + 1 */
            diff_Q15 = silk_SUB_LSHIFT32( in_Q15[ m + 1 ], (opus_int32)cb_Q8_ptr[ m + 1 ], 7 ); /* range: [ -32767 : 32767 ]*/
            diffw_Q24 = silk_SMULBB( diff_Q15, w_Q9_ptr[ m + 1 ] );
            sum_error_Q24 = silk_ADD32( sum_error_Q24, silk_abs( silk_SUB_RSHIFT32( diffw_Q24, pred_Q24, 1 ) ) );
            pred_Q24 = diffw_Q24;

            /* Compute weighted absolute predictive quantization error for index m */
            diff_Q15 = silk_SUB_LSHIFT32( in_Q15[ m ], (opus_int32)cb_Q8_ptr[ m ], 7 ); /* range: [ -32767 : 32767 ]*/
            diffw_Q24 = silk_SMULBB( diff_Q15, w_Q9_ptr[ m ] );
            sum_error_Q24 = silk_ADD32( sum_error_Q24, silk_abs( silk_SUB_RSHIFT32( diffw_Q24, pred_Q24, 1 ) ) );
            pred_Q24 = diffw_Q24;

            silk_assert( sum_error_Q24 >= 0 );
        }
        err_Q24[ i ] = sum_error_Q24;
        cb_Q8_ptr += LPC_order;
        w_Q9_ptr += LPC_order;
    }
}
