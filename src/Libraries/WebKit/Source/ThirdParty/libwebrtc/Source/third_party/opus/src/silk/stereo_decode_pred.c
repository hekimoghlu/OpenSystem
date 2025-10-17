/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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

/* Decode mid/side predictors */
void silk_stereo_decode_pred(
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int32                  pred_Q13[]                      /* O    Predictors                                  */
)
{
    opus_int   n, ix[ 2 ][ 3 ];
    opus_int32 low_Q13, step_Q13;

    /* Entropy decoding */
    n = ec_dec_icdf( psRangeDec, silk_stereo_pred_joint_iCDF, 8 );
    ix[ 0 ][ 2 ] = silk_DIV32_16( n, 5 );
    ix[ 1 ][ 2 ] = n - 5 * ix[ 0 ][ 2 ];
    for( n = 0; n < 2; n++ ) {
        ix[ n ][ 0 ] = ec_dec_icdf( psRangeDec, silk_uniform3_iCDF, 8 );
        ix[ n ][ 1 ] = ec_dec_icdf( psRangeDec, silk_uniform5_iCDF, 8 );
    }

    /* Dequantize */
    for( n = 0; n < 2; n++ ) {
        ix[ n ][ 0 ] += 3 * ix[ n ][ 2 ];
        low_Q13 = silk_stereo_pred_quant_Q13[ ix[ n ][ 0 ] ];
        step_Q13 = silk_SMULWB( silk_stereo_pred_quant_Q13[ ix[ n ][ 0 ] + 1 ] - low_Q13,
            SILK_FIX_CONST( 0.5 / STEREO_QUANT_SUB_STEPS, 16 ) );
        pred_Q13[ n ] = silk_SMLABB( low_Q13, step_Q13, 2 * ix[ n ][ 1 ] + 1 );
    }

    /* Subtract second from first predictor (helps when actually applying these) */
    pred_Q13[ 0 ] -= pred_Q13[ 1 ];
}

/* Decode mid-only flag */
void silk_stereo_decode_mid_only(
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int                    *decode_only_mid                /* O    Flag that only mid channel has been coded   */
)
{
    /* Decode flag that only mid channel is coded */
    *decode_only_mid = ec_dec_icdf( psRangeDec, silk_stereo_only_code_mid_iCDF, 8 );
}
