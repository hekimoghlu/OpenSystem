/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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

#include "define.h"
#include "SigProc_FIX.h"

/*
R. Laroia, N. Phamdo and N. Farvardin, "Robust and Efficient Quantization of Speech LSP
Parameters Using Structured Vector Quantization", Proc. IEEE Int. Conf. Acoust., Speech,
Signal Processing, pp. 641-644, 1991.
*/

/* Laroia low complexity NLSF weights */
void silk_NLSF_VQ_weights_laroia(
    opus_int16                  *pNLSFW_Q_OUT,      /* O     Pointer to input vector weights [D]                        */
    const opus_int16            *pNLSF_Q15,         /* I     Pointer to input vector         [D]                        */
    const opus_int              D                   /* I     Input vector dimension (even)                              */
)
{
    opus_int   k;
    opus_int32 tmp1_int, tmp2_int;

    celt_assert( D > 0 );
    celt_assert( ( D & 1 ) == 0 );

    /* First value */
    tmp1_int = silk_max_int( pNLSF_Q15[ 0 ], 1 );
    tmp1_int = silk_DIV32_16( (opus_int32)1 << ( 15 + NLSF_W_Q ), tmp1_int );
    tmp2_int = silk_max_int( pNLSF_Q15[ 1 ] - pNLSF_Q15[ 0 ], 1 );
    tmp2_int = silk_DIV32_16( (opus_int32)1 << ( 15 + NLSF_W_Q ), tmp2_int );
    pNLSFW_Q_OUT[ 0 ] = (opus_int16)silk_min_int( tmp1_int + tmp2_int, silk_int16_MAX );
    silk_assert( pNLSFW_Q_OUT[ 0 ] > 0 );

    /* Main loop */
    for( k = 1; k < D - 1; k += 2 ) {
        tmp1_int = silk_max_int( pNLSF_Q15[ k + 1 ] - pNLSF_Q15[ k ], 1 );
        tmp1_int = silk_DIV32_16( (opus_int32)1 << ( 15 + NLSF_W_Q ), tmp1_int );
        pNLSFW_Q_OUT[ k ] = (opus_int16)silk_min_int( tmp1_int + tmp2_int, silk_int16_MAX );
        silk_assert( pNLSFW_Q_OUT[ k ] > 0 );

        tmp2_int = silk_max_int( pNLSF_Q15[ k + 2 ] - pNLSF_Q15[ k + 1 ], 1 );
        tmp2_int = silk_DIV32_16( (opus_int32)1 << ( 15 + NLSF_W_Q ), tmp2_int );
        pNLSFW_Q_OUT[ k + 1 ] = (opus_int16)silk_min_int( tmp1_int + tmp2_int, silk_int16_MAX );
        silk_assert( pNLSFW_Q_OUT[ k + 1 ] > 0 );
    }

    /* Last value */
    tmp1_int = silk_max_int( ( 1 << 15 ) - pNLSF_Q15[ D - 1 ], 1 );
    tmp1_int = silk_DIV32_16( (opus_int32)1 << ( 15 + NLSF_W_Q ), tmp1_int );
    pNLSFW_Q_OUT[ D - 1 ] = (opus_int16)silk_min_int( tmp1_int + tmp2_int, silk_int16_MAX );
    silk_assert( pNLSFW_Q_OUT[ D - 1 ] > 0 );
}
