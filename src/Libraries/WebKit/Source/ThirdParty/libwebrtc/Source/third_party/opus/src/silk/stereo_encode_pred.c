/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

/* Entropy code the mid/side quantization indices */
void silk_stereo_encode_pred(
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int8                   ix[ 2 ][ 3 ]                    /* I    Quantization indices                        */
)
{
    opus_int   n;

    /* Entropy coding */
    n = 5 * ix[ 0 ][ 2 ] + ix[ 1 ][ 2 ];
    celt_assert( n < 25 );
    ec_enc_icdf( psRangeEnc, n, silk_stereo_pred_joint_iCDF, 8 );
    for( n = 0; n < 2; n++ ) {
        celt_assert( ix[ n ][ 0 ] < 3 );
        celt_assert( ix[ n ][ 1 ] < STEREO_QUANT_SUB_STEPS );
        ec_enc_icdf( psRangeEnc, ix[ n ][ 0 ], silk_uniform3_iCDF, 8 );
        ec_enc_icdf( psRangeEnc, ix[ n ][ 1 ], silk_uniform5_iCDF, 8 );
    }
}

/* Entropy code the mid-only flag */
void silk_stereo_encode_mid_only(
    ec_enc                      *psRangeEnc,                    /* I/O  Compressor data structure                   */
    opus_int8                   mid_only_flag
)
{
    /* Encode flag that only mid channel is coded */
    ec_enc_icdf( psRangeEnc, mid_only_flag, silk_stereo_only_code_mid_iCDF, 8 );
}
