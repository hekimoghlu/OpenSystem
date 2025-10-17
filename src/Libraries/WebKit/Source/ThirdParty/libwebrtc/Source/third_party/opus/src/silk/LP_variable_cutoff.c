/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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

/*
    Elliptic/Cauer filters designed with 0.1 dB passband ripple,
    80 dB minimum stopband attenuation, and
    [0.95 : 0.15 : 0.35] normalized cut off frequencies.
*/

#include "main.h"

/* Helper function, interpolates the filter taps */
static OPUS_INLINE void silk_LP_interpolate_filter_taps(
    opus_int32           B_Q28[ TRANSITION_NB ],
    opus_int32           A_Q28[ TRANSITION_NA ],
    const opus_int       ind,
    const opus_int32     fac_Q16
)
{
    opus_int nb, na;

    if( ind < TRANSITION_INT_NUM - 1 ) {
        if( fac_Q16 > 0 ) {
            if( fac_Q16 < 32768 ) { /* fac_Q16 is in range of a 16-bit int */
                /* Piece-wise linear interpolation of B and A */
                for( nb = 0; nb < TRANSITION_NB; nb++ ) {
                    B_Q28[ nb ] = silk_SMLAWB(
                        silk_Transition_LP_B_Q28[ ind     ][ nb ],
                        silk_Transition_LP_B_Q28[ ind + 1 ][ nb ] -
                        silk_Transition_LP_B_Q28[ ind     ][ nb ],
                        fac_Q16 );
                }
                for( na = 0; na < TRANSITION_NA; na++ ) {
                    A_Q28[ na ] = silk_SMLAWB(
                        silk_Transition_LP_A_Q28[ ind     ][ na ],
                        silk_Transition_LP_A_Q28[ ind + 1 ][ na ] -
                        silk_Transition_LP_A_Q28[ ind     ][ na ],
                        fac_Q16 );
                }
            } else { /* ( fac_Q16 - ( 1 << 16 ) ) is in range of a 16-bit int */
                silk_assert( fac_Q16 - ( 1 << 16 ) == silk_SAT16( fac_Q16 - ( 1 << 16 ) ) );
                /* Piece-wise linear interpolation of B and A */
                for( nb = 0; nb < TRANSITION_NB; nb++ ) {
                    B_Q28[ nb ] = silk_SMLAWB(
                        silk_Transition_LP_B_Q28[ ind + 1 ][ nb ],
                        silk_Transition_LP_B_Q28[ ind + 1 ][ nb ] -
                        silk_Transition_LP_B_Q28[ ind     ][ nb ],
                        fac_Q16 - ( (opus_int32)1 << 16 ) );
                }
                for( na = 0; na < TRANSITION_NA; na++ ) {
                    A_Q28[ na ] = silk_SMLAWB(
                        silk_Transition_LP_A_Q28[ ind + 1 ][ na ],
                        silk_Transition_LP_A_Q28[ ind + 1 ][ na ] -
                        silk_Transition_LP_A_Q28[ ind     ][ na ],
                        fac_Q16 - ( (opus_int32)1 << 16 ) );
                }
            }
        } else {
            silk_memcpy( B_Q28, silk_Transition_LP_B_Q28[ ind ], TRANSITION_NB * sizeof( opus_int32 ) );
            silk_memcpy( A_Q28, silk_Transition_LP_A_Q28[ ind ], TRANSITION_NA * sizeof( opus_int32 ) );
        }
    } else {
        silk_memcpy( B_Q28, silk_Transition_LP_B_Q28[ TRANSITION_INT_NUM - 1 ], TRANSITION_NB * sizeof( opus_int32 ) );
        silk_memcpy( A_Q28, silk_Transition_LP_A_Q28[ TRANSITION_INT_NUM - 1 ], TRANSITION_NA * sizeof( opus_int32 ) );
    }
}

/* Low-pass filter with variable cutoff frequency based on  */
/* piece-wise linear interpolation between elliptic filters */
/* Start by setting psEncC->mode <> 0;                      */
/* Deactivate by setting psEncC->mode = 0;                  */
void silk_LP_variable_cutoff(
    silk_LP_state               *psLP,                          /* I/O  LP filter state                             */
    opus_int16                  *frame,                         /* I/O  Low-pass filtered output signal             */
    const opus_int              frame_length                    /* I    Frame length                                */
)
{
    opus_int32   B_Q28[ TRANSITION_NB ], A_Q28[ TRANSITION_NA ], fac_Q16 = 0;
    opus_int     ind = 0;

    silk_assert( psLP->transition_frame_no >= 0 && psLP->transition_frame_no <= TRANSITION_FRAMES );

    /* Run filter if needed */
    if( psLP->mode != 0 ) {
        /* Calculate index and interpolation factor for interpolation */
#if( TRANSITION_INT_STEPS == 64 )
        fac_Q16 = silk_LSHIFT( TRANSITION_FRAMES - psLP->transition_frame_no, 16 - 6 );
#else
        fac_Q16 = silk_DIV32_16( silk_LSHIFT( TRANSITION_FRAMES - psLP->transition_frame_no, 16 ), TRANSITION_FRAMES );
#endif
        ind      = silk_RSHIFT( fac_Q16, 16 );
        fac_Q16 -= silk_LSHIFT( ind, 16 );

        silk_assert( ind >= 0 );
        silk_assert( ind < TRANSITION_INT_NUM );

        /* Interpolate filter coefficients */
        silk_LP_interpolate_filter_taps( B_Q28, A_Q28, ind, fac_Q16 );

        /* Update transition frame number for next frame */
        psLP->transition_frame_no = silk_LIMIT( psLP->transition_frame_no + psLP->mode, 0, TRANSITION_FRAMES );

        /* ARMA low-pass filtering */
        silk_assert( TRANSITION_NB == 3 && TRANSITION_NA == 2 );
        silk_biquad_alt_stride1( frame, B_Q28, A_Q28, psLP->In_LP_State, frame, frame_length);
    }
}
