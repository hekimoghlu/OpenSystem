/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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

/* Quantize mid/side predictors */
void silk_stereo_quant_pred(
    opus_int32                  pred_Q13[],                     /* I/O  Predictors (out: quantized)                 */
    opus_int8                   ix[ 2 ][ 3 ]                    /* O    Quantization indices                        */
)
{
    opus_int   i, j, n;
    opus_int32 low_Q13, step_Q13, lvl_Q13, err_min_Q13, err_Q13, quant_pred_Q13 = 0;

    /* Quantize */
    for( n = 0; n < 2; n++ ) {
        /* Brute-force search over quantization levels */
        err_min_Q13 = silk_int32_MAX;
        for( i = 0; i < STEREO_QUANT_TAB_SIZE - 1; i++ ) {
            low_Q13 = silk_stereo_pred_quant_Q13[ i ];
            step_Q13 = silk_SMULWB( silk_stereo_pred_quant_Q13[ i + 1 ] - low_Q13,
                SILK_FIX_CONST( 0.5 / STEREO_QUANT_SUB_STEPS, 16 ) );
            for( j = 0; j < STEREO_QUANT_SUB_STEPS; j++ ) {
                lvl_Q13 = silk_SMLABB( low_Q13, step_Q13, 2 * j + 1 );
                err_Q13 = silk_abs( pred_Q13[ n ] - lvl_Q13 );
                if( err_Q13 < err_min_Q13 ) {
                    err_min_Q13 = err_Q13;
                    quant_pred_Q13 = lvl_Q13;
                    ix[ n ][ 0 ] = i;
                    ix[ n ][ 1 ] = j;
                } else {
                    /* Error increasing, so we're past the optimum */
                    goto done;
                }
            }
        }
        done:
        ix[ n ][ 2 ]  = silk_DIV32_16( ix[ n ][ 0 ], 3 );
        ix[ n ][ 0 ] -= ix[ n ][ 2 ] * 3;
        pred_Q13[ n ] = quant_pred_Q13;
    }

    /* Subtract second from first predictor (helps when actually applying these) */
    pred_Q13[ 0 ] -= pred_Q13[ 1 ];
}
