/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

/*********************************************/
/* Decode quantization indices of excitation */
/*********************************************/
void silk_decode_pulses(
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int16                  pulses[],                       /* O    Excitation signal                           */
    const opus_int              signalType,                     /* I    Sigtype                                     */
    const opus_int              quantOffsetType,                /* I    quantOffsetType                             */
    const opus_int              frame_length                    /* I    Frame length                                */
)
{
    opus_int   i, j, k, iter, abs_q, nLS, RateLevelIndex;
    opus_int   sum_pulses[ MAX_NB_SHELL_BLOCKS ], nLshifts[ MAX_NB_SHELL_BLOCKS ];
    opus_int16 *pulses_ptr;
    const opus_uint8 *cdf_ptr;

    /*********************/
    /* Decode rate level */
    /*********************/
    RateLevelIndex = ec_dec_icdf( psRangeDec, silk_rate_levels_iCDF[ signalType >> 1 ], 8 );

    /* Calculate number of shell blocks */
    silk_assert( 1 << LOG2_SHELL_CODEC_FRAME_LENGTH == SHELL_CODEC_FRAME_LENGTH );
    iter = silk_RSHIFT( frame_length, LOG2_SHELL_CODEC_FRAME_LENGTH );
    if( iter * SHELL_CODEC_FRAME_LENGTH < frame_length ) {
        celt_assert( frame_length == 12 * 10 ); /* Make sure only happens for 10 ms @ 12 kHz */
        iter++;
    }

    /***************************************************/
    /* Sum-Weighted-Pulses Decoding                    */
    /***************************************************/
    cdf_ptr = silk_pulses_per_block_iCDF[ RateLevelIndex ];
    for( i = 0; i < iter; i++ ) {
        nLshifts[ i ] = 0;
        sum_pulses[ i ] = ec_dec_icdf( psRangeDec, cdf_ptr, 8 );

        /* LSB indication */
        while( sum_pulses[ i ] == SILK_MAX_PULSES + 1 ) {
            nLshifts[ i ]++;
            /* When we've already got 10 LSBs, we shift the table to not allow (SILK_MAX_PULSES + 1) */
            sum_pulses[ i ] = ec_dec_icdf( psRangeDec,
                    silk_pulses_per_block_iCDF[ N_RATE_LEVELS - 1] + ( nLshifts[ i ] == 10 ), 8 );
        }
    }

    /***************************************************/
    /* Shell decoding                                  */
    /***************************************************/
    for( i = 0; i < iter; i++ ) {
        if( sum_pulses[ i ] > 0 ) {
            silk_shell_decoder( &pulses[ silk_SMULBB( i, SHELL_CODEC_FRAME_LENGTH ) ], psRangeDec, sum_pulses[ i ] );
        } else {
            silk_memset( &pulses[ silk_SMULBB( i, SHELL_CODEC_FRAME_LENGTH ) ], 0, SHELL_CODEC_FRAME_LENGTH * sizeof( pulses[0] ) );
        }
    }

    /***************************************************/
    /* LSB Decoding                                    */
    /***************************************************/
    for( i = 0; i < iter; i++ ) {
        if( nLshifts[ i ] > 0 ) {
            nLS = nLshifts[ i ];
            pulses_ptr = &pulses[ silk_SMULBB( i, SHELL_CODEC_FRAME_LENGTH ) ];
            for( k = 0; k < SHELL_CODEC_FRAME_LENGTH; k++ ) {
                abs_q = pulses_ptr[ k ];
                for( j = 0; j < nLS; j++ ) {
                    abs_q = silk_LSHIFT( abs_q, 1 );
                    abs_q += ec_dec_icdf( psRangeDec, silk_lsb_iCDF, 8 );
                }
                pulses_ptr[ k ] = abs_q;
            }
            /* Mark the number of pulses non-zero for sign decoding. */
            sum_pulses[ i ] |= nLS << 5;
        }
    }

    /****************************************/
    /* Decode and add signs to pulse signal */
    /****************************************/
    silk_decode_signs( psRangeDec, pulses, frame_length, signalType, quantOffsetType, sum_pulses );
}
