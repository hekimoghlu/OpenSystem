/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#include "resampler_private.h"
#include "stack_alloc.h"

#define ORDER_FIR                   4

/* Downsample by a factor 2/3, low quality */
void silk_resampler_down2_3(
    opus_int32                  *S,                 /* I/O  State vector [ 6 ]                                          */
    opus_int16                  *out,               /* O    Output signal [ floor(2*inLen/3) ]                          */
    const opus_int16            *in,                /* I    Input signal [ inLen ]                                      */
    opus_int32                  inLen               /* I    Number of input samples                                     */
)
{
    opus_int32 nSamplesIn, counter, res_Q6;
    VARDECL( opus_int32, buf );
    opus_int32 *buf_ptr;
    SAVE_STACK;

    ALLOC( buf, RESAMPLER_MAX_BATCH_SIZE_IN + ORDER_FIR, opus_int32 );

    /* Copy buffered samples to start of buffer */
    silk_memcpy( buf, S, ORDER_FIR * sizeof( opus_int32 ) );

    /* Iterate over blocks of frameSizeIn input samples */
    while( 1 ) {
        nSamplesIn = silk_min( inLen, RESAMPLER_MAX_BATCH_SIZE_IN );

        /* Second-order AR filter (output in Q8) */
        silk_resampler_private_AR2( &S[ ORDER_FIR ], &buf[ ORDER_FIR ], in,
            silk_Resampler_2_3_COEFS_LQ, nSamplesIn );

        /* Interpolate filtered signal */
        buf_ptr = buf;
        counter = nSamplesIn;
        while( counter > 2 ) {
            /* Inner product */
            res_Q6 = silk_SMULWB(         buf_ptr[ 0 ], silk_Resampler_2_3_COEFS_LQ[ 2 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 1 ], silk_Resampler_2_3_COEFS_LQ[ 3 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 2 ], silk_Resampler_2_3_COEFS_LQ[ 5 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 3 ], silk_Resampler_2_3_COEFS_LQ[ 4 ] );

            /* Scale down, saturate and store in output array */
            *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );

            res_Q6 = silk_SMULWB(         buf_ptr[ 1 ], silk_Resampler_2_3_COEFS_LQ[ 4 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 2 ], silk_Resampler_2_3_COEFS_LQ[ 5 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 3 ], silk_Resampler_2_3_COEFS_LQ[ 3 ] );
            res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 4 ], silk_Resampler_2_3_COEFS_LQ[ 2 ] );

            /* Scale down, saturate and store in output array */
            *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );

            buf_ptr += 3;
            counter -= 3;
        }

        in += nSamplesIn;
        inLen -= nSamplesIn;

        if( inLen > 0 ) {
            /* More iterations to do; copy last part of filtered signal to beginning of buffer */
            silk_memcpy( buf, &buf[ nSamplesIn ], ORDER_FIR * sizeof( opus_int32 ) );
        } else {
            break;
        }
    }

    /* Copy last part of filtered signal to the state for the next call */
    silk_memcpy( S, &buf[ nSamplesIn ], ORDER_FIR * sizeof( opus_int32 ) );
    RESTORE_STACK;
}
