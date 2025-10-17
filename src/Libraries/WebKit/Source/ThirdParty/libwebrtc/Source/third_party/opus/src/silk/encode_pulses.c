/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "stack_alloc.h"

/*********************************************/
/* Encode quantization indices of excitation */
/*********************************************/

static OPUS_INLINE opus_int combine_and_check(    /* return ok                           */
    opus_int         *pulses_comb,           /* O                                   */
    const opus_int   *pulses_in,             /* I                                   */
    opus_int         max_pulses,             /* I    max value for sum of pulses    */
    opus_int         len                     /* I    number of output values        */
)
{
    opus_int k, sum;

    for( k = 0; k < len; k++ ) {
        sum = pulses_in[ 2 * k ] + pulses_in[ 2 * k + 1 ];
        if( sum > max_pulses ) {
            return 1;
        }
        pulses_comb[ k ] = sum;
    }

    return 0;
}

/* Encode quantization indices of excitation */
void silk_encode_pulses(
    ec_enc                      *psRangeEnc,                    /* I/O  compressor data structure                   */
    const opus_int              signalType,                     /* I    Signal type                                 */
    const opus_int              quantOffsetType,                /* I    quantOffsetType                             */
    opus_int8                   pulses[],                       /* I    quantization indices                        */
    const opus_int              frame_length                    /* I    Frame length                                */
)
{
    opus_int   i, k, j, iter, bit, nLS, scale_down, RateLevelIndex = 0;
    opus_int32 abs_q, minSumBits_Q5, sumBits_Q5;
    VARDECL( opus_int, abs_pulses );
    VARDECL( opus_int, sum_pulses );
    VARDECL( opus_int, nRshifts );
    opus_int   pulses_comb[ 8 ];
    opus_int   *abs_pulses_ptr;
    const opus_int8 *pulses_ptr;
    const opus_uint8 *cdf_ptr;
    const opus_uint8 *nBits_ptr;
    SAVE_STACK;

    silk_memset( pulses_comb, 0, 8 * sizeof( opus_int ) ); /* Fixing Valgrind reported problem*/

    /****************************/
    /* Prepare for shell coding */
    /****************************/
    /* Calculate number of shell blocks */
    silk_assert( 1 << LOG2_SHELL_CODEC_FRAME_LENGTH == SHELL_CODEC_FRAME_LENGTH );
    iter = silk_RSHIFT( frame_length, LOG2_SHELL_CODEC_FRAME_LENGTH );
    if( iter * SHELL_CODEC_FRAME_LENGTH < frame_length ) {
        celt_assert( frame_length == 12 * 10 ); /* Make sure only happens for 10 ms @ 12 kHz */
        iter++;
        silk_memset( &pulses[ frame_length ], 0, SHELL_CODEC_FRAME_LENGTH * sizeof(opus_int8));
    }

    /* Take the absolute value of the pulses */
    ALLOC( abs_pulses, iter * SHELL_CODEC_FRAME_LENGTH, opus_int );
    silk_assert( !( SHELL_CODEC_FRAME_LENGTH & 3 ) );
    for( i = 0; i < iter * SHELL_CODEC_FRAME_LENGTH; i+=4 ) {
        abs_pulses[i+0] = ( opus_int )silk_abs( pulses[ i + 0 ] );
        abs_pulses[i+1] = ( opus_int )silk_abs( pulses[ i + 1 ] );
        abs_pulses[i+2] = ( opus_int )silk_abs( pulses[ i + 2 ] );
        abs_pulses[i+3] = ( opus_int )silk_abs( pulses[ i + 3 ] );
    }

    /* Calc sum pulses per shell code frame */
    ALLOC( sum_pulses, iter, opus_int );
    ALLOC( nRshifts, iter, opus_int );
    abs_pulses_ptr = abs_pulses;
    for( i = 0; i < iter; i++ ) {
        nRshifts[ i ] = 0;

        while( 1 ) {
            /* 1+1 -> 2 */
            scale_down = combine_and_check( pulses_comb, abs_pulses_ptr, silk_max_pulses_table[ 0 ], 8 );
            /* 2+2 -> 4 */
            scale_down += combine_and_check( pulses_comb, pulses_comb, silk_max_pulses_table[ 1 ], 4 );
            /* 4+4 -> 8 */
            scale_down += combine_and_check( pulses_comb, pulses_comb, silk_max_pulses_table[ 2 ], 2 );
            /* 8+8 -> 16 */
            scale_down += combine_and_check( &sum_pulses[ i ], pulses_comb, silk_max_pulses_table[ 3 ], 1 );

            if( scale_down ) {
                /* We need to downscale the quantization signal */
                nRshifts[ i ]++;
                for( k = 0; k < SHELL_CODEC_FRAME_LENGTH; k++ ) {
                    abs_pulses_ptr[ k ] = silk_RSHIFT( abs_pulses_ptr[ k ], 1 );
                }
            } else {
                /* Jump out of while(1) loop and go to next shell coding frame */
                break;
            }
        }
        abs_pulses_ptr += SHELL_CODEC_FRAME_LENGTH;
    }

    /**************/
    /* Rate level */
    /**************/
    /* find rate level that leads to fewest bits for coding of pulses per block info */
    minSumBits_Q5 = silk_int32_MAX;
    for( k = 0; k < N_RATE_LEVELS - 1; k++ ) {
        nBits_ptr  = silk_pulses_per_block_BITS_Q5[ k ];
        sumBits_Q5 = silk_rate_levels_BITS_Q5[ signalType >> 1 ][ k ];
        for( i = 0; i < iter; i++ ) {
            if( nRshifts[ i ] > 0 ) {
                sumBits_Q5 += nBits_ptr[ SILK_MAX_PULSES + 1 ];
            } else {
                sumBits_Q5 += nBits_ptr[ sum_pulses[ i ] ];
            }
        }
        if( sumBits_Q5 < minSumBits_Q5 ) {
            minSumBits_Q5 = sumBits_Q5;
            RateLevelIndex = k;
        }
    }
    ec_enc_icdf( psRangeEnc, RateLevelIndex, silk_rate_levels_iCDF[ signalType >> 1 ], 8 );

    /***************************************************/
    /* Sum-Weighted-Pulses Encoding                    */
    /***************************************************/
    cdf_ptr = silk_pulses_per_block_iCDF[ RateLevelIndex ];
    for( i = 0; i < iter; i++ ) {
        if( nRshifts[ i ] == 0 ) {
            ec_enc_icdf( psRangeEnc, sum_pulses[ i ], cdf_ptr, 8 );
        } else {
            ec_enc_icdf( psRangeEnc, SILK_MAX_PULSES + 1, cdf_ptr, 8 );
            for( k = 0; k < nRshifts[ i ] - 1; k++ ) {
                ec_enc_icdf( psRangeEnc, SILK_MAX_PULSES + 1, silk_pulses_per_block_iCDF[ N_RATE_LEVELS - 1 ], 8 );
            }
            ec_enc_icdf( psRangeEnc, sum_pulses[ i ], silk_pulses_per_block_iCDF[ N_RATE_LEVELS - 1 ], 8 );
        }
    }

    /******************/
    /* Shell Encoding */
    /******************/
    for( i = 0; i < iter; i++ ) {
        if( sum_pulses[ i ] > 0 ) {
            silk_shell_encoder( psRangeEnc, &abs_pulses[ i * SHELL_CODEC_FRAME_LENGTH ] );
        }
    }

    /****************/
    /* LSB Encoding */
    /****************/
    for( i = 0; i < iter; i++ ) {
        if( nRshifts[ i ] > 0 ) {
            pulses_ptr = &pulses[ i * SHELL_CODEC_FRAME_LENGTH ];
            nLS = nRshifts[ i ] - 1;
            for( k = 0; k < SHELL_CODEC_FRAME_LENGTH; k++ ) {
                abs_q = (opus_int8)silk_abs( pulses_ptr[ k ] );
                for( j = nLS; j > 0; j-- ) {
                    bit = silk_RSHIFT( abs_q, j ) & 1;
                    ec_enc_icdf( psRangeEnc, bit, silk_lsb_iCDF, 8 );
                }
                bit = abs_q & 1;
                ec_enc_icdf( psRangeEnc, bit, silk_lsb_iCDF, 8 );
            }
        }
    }

    /****************/
    /* Encode signs */
    /****************/
    silk_encode_signs( psRangeEnc, pulses, frame_length, signalType, quantOffsetType, sum_pulses );
    RESTORE_STACK;
}
