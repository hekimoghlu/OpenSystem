/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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

/* Convert adaptive Mid/Side representation to Left/Right stereo signal */
void silk_stereo_MS_to_LR(
    stereo_dec_state            *state,                         /* I/O  State                                       */
    opus_int16                  x1[],                           /* I/O  Left input signal, becomes mid signal       */
    opus_int16                  x2[],                           /* I/O  Right input signal, becomes side signal     */
    const opus_int32            pred_Q13[],                     /* I    Predictors                                  */
    opus_int                    fs_kHz,                         /* I    Samples rate (kHz)                          */
    opus_int                    frame_length                    /* I    Number of samples                           */
)
{
    opus_int   n, denom_Q16, delta0_Q13, delta1_Q13;
    opus_int32 sum, diff, pred0_Q13, pred1_Q13;

    /* Buffering */
    silk_memcpy( x1, state->sMid,  2 * sizeof( opus_int16 ) );
    silk_memcpy( x2, state->sSide, 2 * sizeof( opus_int16 ) );
    silk_memcpy( state->sMid,  &x1[ frame_length ], 2 * sizeof( opus_int16 ) );
    silk_memcpy( state->sSide, &x2[ frame_length ], 2 * sizeof( opus_int16 ) );

    /* Interpolate predictors and add prediction to side channel */
    pred0_Q13  = state->pred_prev_Q13[ 0 ];
    pred1_Q13  = state->pred_prev_Q13[ 1 ];
    denom_Q16  = silk_DIV32_16( (opus_int32)1 << 16, STEREO_INTERP_LEN_MS * fs_kHz );
    delta0_Q13 = silk_RSHIFT_ROUND( silk_SMULBB( pred_Q13[ 0 ] - state->pred_prev_Q13[ 0 ], denom_Q16 ), 16 );
    delta1_Q13 = silk_RSHIFT_ROUND( silk_SMULBB( pred_Q13[ 1 ] - state->pred_prev_Q13[ 1 ], denom_Q16 ), 16 );
    for( n = 0; n < STEREO_INTERP_LEN_MS * fs_kHz; n++ ) {
        pred0_Q13 += delta0_Q13;
        pred1_Q13 += delta1_Q13;
        sum = silk_LSHIFT( silk_ADD_LSHIFT32( x1[ n ] + (opus_int32)x1[ n + 2 ], x1[ n + 1 ], 1 ), 9 );       /* Q11 */
        sum = silk_SMLAWB( silk_LSHIFT( (opus_int32)x2[ n + 1 ], 8 ), sum, pred0_Q13 );         /* Q8  */
        sum = silk_SMLAWB( sum, silk_LSHIFT( (opus_int32)x1[ n + 1 ], 11 ), pred1_Q13 );        /* Q8  */
        x2[ n + 1 ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( sum, 8 ) );
    }
    pred0_Q13 = pred_Q13[ 0 ];
    pred1_Q13 = pred_Q13[ 1 ];
    for( n = STEREO_INTERP_LEN_MS * fs_kHz; n < frame_length; n++ ) {
        sum = silk_LSHIFT( silk_ADD_LSHIFT32( x1[ n ] + (opus_int32)x1[ n + 2 ], x1[ n + 1 ], 1 ), 9 );       /* Q11 */
        sum = silk_SMLAWB( silk_LSHIFT( (opus_int32)x2[ n + 1 ], 8 ), sum, pred0_Q13 );         /* Q8  */
        sum = silk_SMLAWB( sum, silk_LSHIFT( (opus_int32)x1[ n + 1 ], 11 ), pred1_Q13 );        /* Q8  */
        x2[ n + 1 ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( sum, 8 ) );
    }
    state->pred_prev_Q13[ 0 ] = pred_Q13[ 0 ];
    state->pred_prev_Q13[ 1 ] = pred_Q13[ 1 ];

    /* Convert to left/right signals */
    for( n = 0; n < frame_length; n++ ) {
        sum  = x1[ n + 1 ] + (opus_int32)x2[ n + 1 ];
        diff = x1[ n + 1 ] - (opus_int32)x2[ n + 1 ];
        x1[ n + 1 ] = (opus_int16)silk_SAT16( sum );
        x2[ n + 1 ] = (opus_int16)silk_SAT16( diff );
    }
}
