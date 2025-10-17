/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include "resampler_rom.h"

/* Downsample by a factor 2 */
void silk_resampler_down2(
    opus_int32                  *S,                 /* I/O  State vector [ 2 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ floor(len/2) ]                              */
    const opus_int16            *in,                /* I    Input signal [ len ]                                        */
    opus_int32                  inLen               /* I    Number of input samples                                     */
)
{
    opus_int32 k, len2 = silk_RSHIFT32( inLen, 1 );
    opus_int32 in32, out32, Y, X;

    celt_assert( silk_resampler_down2_0 > 0 );
    celt_assert( silk_resampler_down2_1 < 0 );

    /* Internal variables and state are in Q10 format */
    for( k = 0; k < len2; k++ ) {
        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k ], 10 );

        /* All-pass section for even input sample */
        Y      = silk_SUB32( in32, S[ 0 ] );
        X      = silk_SMLAWB( Y, Y, silk_resampler_down2_1 );
        out32  = silk_ADD32( S[ 0 ], X );
        S[ 0 ] = silk_ADD32( in32, X );

        /* Convert to Q10 */
        in32 = silk_LSHIFT( (opus_int32)in[ 2 * k + 1 ], 10 );

        /* All-pass section for odd input sample, and add to output of previous section */
        Y      = silk_SUB32( in32, S[ 1 ] );
        X      = silk_SMULWB( Y, silk_resampler_down2_0 );
        out32  = silk_ADD32( out32, S[ 1 ] );
        out32  = silk_ADD32( out32, X );
        S[ 1 ] = silk_ADD32( in32, X );

        /* Add, convert back to int16 and store to output */
        out[ k ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( out32, 11 ) );
    }
}

