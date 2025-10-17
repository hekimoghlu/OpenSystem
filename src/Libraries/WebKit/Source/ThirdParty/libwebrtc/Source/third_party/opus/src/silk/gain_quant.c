/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

#define OFFSET                  ( ( MIN_QGAIN_DB * 128 ) / 6 + 16 * 128 )
#define SCALE_Q16               ( ( 65536 * ( N_LEVELS_QGAIN - 1 ) ) / ( ( ( MAX_QGAIN_DB - MIN_QGAIN_DB ) * 128 ) / 6 ) )
#define INV_SCALE_Q16           ( ( 65536 * ( ( ( MAX_QGAIN_DB - MIN_QGAIN_DB ) * 128 ) / 6 ) ) / ( N_LEVELS_QGAIN - 1 ) )

/* Gain scalar quantization with hysteresis, uniform on log scale */
void silk_gains_quant(
    opus_int8                   ind[ MAX_NB_SUBFR ],            /* O    gain indices                                */
    opus_int32                  gain_Q16[ MAX_NB_SUBFR ],       /* I/O  gains (quantized out)                       */
    opus_int8                   *prev_ind,                      /* I/O  last index in previous frame                */
    const opus_int              conditional,                    /* I    first gain is delta coded if 1              */
    const opus_int              nb_subfr                        /* I    number of subframes                         */
)
{
    opus_int k, double_step_size_threshold;

    for( k = 0; k < nb_subfr; k++ ) {
        /* Convert to log scale, scale, floor() */
        ind[ k ] = silk_SMULWB( SCALE_Q16, silk_lin2log( gain_Q16[ k ] ) - OFFSET );

        /* Round towards previous quantized gain (hysteresis) */
        if( ind[ k ] < *prev_ind ) {
            ind[ k ]++;
        }
        ind[ k ] = silk_LIMIT_int( ind[ k ], 0, N_LEVELS_QGAIN - 1 );

        /* Compute delta indices and limit */
        if( k == 0 && conditional == 0 ) {
            /* Full index */
            ind[ k ] = silk_LIMIT_int( ind[ k ], *prev_ind + MIN_DELTA_GAIN_QUANT, N_LEVELS_QGAIN - 1 );
            *prev_ind = ind[ k ];
        } else {
            /* Delta index */
            ind[ k ] = ind[ k ] - *prev_ind;

            /* Double the quantization step size for large gain increases, so that the max gain level can be reached */
            double_step_size_threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind;
            if( ind[ k ] > double_step_size_threshold ) {
                ind[ k ] = double_step_size_threshold + silk_RSHIFT( ind[ k ] - double_step_size_threshold + 1, 1 );
            }

            ind[ k ] = silk_LIMIT_int( ind[ k ], MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT );

            /* Accumulate deltas */
            if( ind[ k ] > double_step_size_threshold ) {
                *prev_ind += silk_LSHIFT( ind[ k ], 1 ) - double_step_size_threshold;
                *prev_ind = silk_min_int( *prev_ind, N_LEVELS_QGAIN - 1 );
            } else {
                *prev_ind += ind[ k ];
            }

            /* Shift to make non-negative */
            ind[ k ] -= MIN_DELTA_GAIN_QUANT;
        }

        /* Scale and convert to linear scale */
        gain_Q16[ k ] = silk_log2lin( silk_min_32( silk_SMULWB( INV_SCALE_Q16, *prev_ind ) + OFFSET, 3967 ) ); /* 3967 = 31 in Q7 */
    }
}

/* Gains scalar dequantization, uniform on log scale */
void silk_gains_dequant(
    opus_int32                  gain_Q16[ MAX_NB_SUBFR ],       /* O    quantized gains                             */
    const opus_int8             ind[ MAX_NB_SUBFR ],            /* I    gain indices                                */
    opus_int8                   *prev_ind,                      /* I/O  last index in previous frame                */
    const opus_int              conditional,                    /* I    first gain is delta coded if 1              */
    const opus_int              nb_subfr                        /* I    number of subframes                          */
)
{
    opus_int   k, ind_tmp, double_step_size_threshold;

    for( k = 0; k < nb_subfr; k++ ) {
        if( k == 0 && conditional == 0 ) {
            /* Gain index is not allowed to go down more than 16 steps (~21.8 dB) */
            *prev_ind = silk_max_int( ind[ k ], *prev_ind - 16 );
        } else {
            /* Delta index */
            ind_tmp = ind[ k ] + MIN_DELTA_GAIN_QUANT;

            /* Accumulate deltas */
            double_step_size_threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind;
            if( ind_tmp > double_step_size_threshold ) {
                *prev_ind += silk_LSHIFT( ind_tmp, 1 ) - double_step_size_threshold;
            } else {
                *prev_ind += ind_tmp;
            }
        }
        *prev_ind = silk_LIMIT_int( *prev_ind, 0, N_LEVELS_QGAIN - 1 );

        /* Scale and convert to linear scale */
        gain_Q16[ k ] = silk_log2lin( silk_min_32( silk_SMULWB( INV_SCALE_Q16, *prev_ind ) + OFFSET, 3967 ) ); /* 3967 = 31 in Q7 */
    }
}

/* Compute unique identifier of gain indices vector */
opus_int32 silk_gains_ID(                                       /* O    returns unique identifier of gains          */
    const opus_int8             ind[ MAX_NB_SUBFR ],            /* I    gain indices                                */
    const opus_int              nb_subfr                        /* I    number of subframes                         */
)
{
    opus_int   k;
    opus_int32 gainsID;

    gainsID = 0;
    for( k = 0; k < nb_subfr; k++ ) {
        gainsID = silk_ADD_LSHIFT32( ind[ k ], gainsID, 8 );
    }

    return gainsID;
}
