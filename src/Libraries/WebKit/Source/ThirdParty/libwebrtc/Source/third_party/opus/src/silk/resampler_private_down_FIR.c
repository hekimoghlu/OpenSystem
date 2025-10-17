/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 17, 2025.
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

static OPUS_INLINE opus_int16 *silk_resampler_private_down_FIR_INTERPOL(
    opus_int16          *out,
    opus_int32          *buf,
    const opus_int16    *FIR_Coefs,
    opus_int            FIR_Order,
    opus_int            FIR_Fracs,
    opus_int32          max_index_Q16,
    opus_int32          index_increment_Q16
)
{
    opus_int32 index_Q16, res_Q6;
    opus_int32 *buf_ptr;
    opus_int32 interpol_ind;
    const opus_int16 *interpol_ptr;

    switch( FIR_Order ) {
        case RESAMPLER_DOWN_ORDER_FIR0:
            for( index_Q16 = 0; index_Q16 < max_index_Q16; index_Q16 += index_increment_Q16 ) {
                /* Integer part gives pointer to buffered input */
                buf_ptr = buf + silk_RSHIFT( index_Q16, 16 );

                /* Fractional part gives interpolation coefficients */
                interpol_ind = silk_SMULWB( index_Q16 & 0xFFFF, FIR_Fracs );

                /* Inner product */
                interpol_ptr = &FIR_Coefs[ RESAMPLER_DOWN_ORDER_FIR0 / 2 * interpol_ind ];
                res_Q6 = silk_SMULWB(         buf_ptr[ 0 ], interpol_ptr[ 0 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 1 ], interpol_ptr[ 1 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 2 ], interpol_ptr[ 2 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 3 ], interpol_ptr[ 3 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 4 ], interpol_ptr[ 4 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 5 ], interpol_ptr[ 5 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 6 ], interpol_ptr[ 6 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 7 ], interpol_ptr[ 7 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 8 ], interpol_ptr[ 8 ] );
                interpol_ptr = &FIR_Coefs[ RESAMPLER_DOWN_ORDER_FIR0 / 2 * ( FIR_Fracs - 1 - interpol_ind ) ];
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 17 ], interpol_ptr[ 0 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 16 ], interpol_ptr[ 1 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 15 ], interpol_ptr[ 2 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 14 ], interpol_ptr[ 3 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 13 ], interpol_ptr[ 4 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 12 ], interpol_ptr[ 5 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 11 ], interpol_ptr[ 6 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[ 10 ], interpol_ptr[ 7 ] );
                res_Q6 = silk_SMLAWB( res_Q6, buf_ptr[  9 ], interpol_ptr[ 8 ] );

                /* Scale down, saturate and store in output array */
                *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );
            }
            break;
        case RESAMPLER_DOWN_ORDER_FIR1:
            for( index_Q16 = 0; index_Q16 < max_index_Q16; index_Q16 += index_increment_Q16 ) {
                /* Integer part gives pointer to buffered input */
                buf_ptr = buf + silk_RSHIFT( index_Q16, 16 );

                /* Inner product */
                res_Q6 = silk_SMULWB(         silk_ADD32( buf_ptr[  0 ], buf_ptr[ 23 ] ), FIR_Coefs[  0 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  1 ], buf_ptr[ 22 ] ), FIR_Coefs[  1 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  2 ], buf_ptr[ 21 ] ), FIR_Coefs[  2 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  3 ], buf_ptr[ 20 ] ), FIR_Coefs[  3 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  4 ], buf_ptr[ 19 ] ), FIR_Coefs[  4 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  5 ], buf_ptr[ 18 ] ), FIR_Coefs[  5 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  6 ], buf_ptr[ 17 ] ), FIR_Coefs[  6 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  7 ], buf_ptr[ 16 ] ), FIR_Coefs[  7 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  8 ], buf_ptr[ 15 ] ), FIR_Coefs[  8 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  9 ], buf_ptr[ 14 ] ), FIR_Coefs[  9 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 10 ], buf_ptr[ 13 ] ), FIR_Coefs[ 10 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 11 ], buf_ptr[ 12 ] ), FIR_Coefs[ 11 ] );

                /* Scale down, saturate and store in output array */
                *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );
            }
            break;
        case RESAMPLER_DOWN_ORDER_FIR2:
            for( index_Q16 = 0; index_Q16 < max_index_Q16; index_Q16 += index_increment_Q16 ) {
                /* Integer part gives pointer to buffered input */
                buf_ptr = buf + silk_RSHIFT( index_Q16, 16 );

                /* Inner product */
                res_Q6 = silk_SMULWB(         silk_ADD32( buf_ptr[  0 ], buf_ptr[ 35 ] ), FIR_Coefs[  0 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  1 ], buf_ptr[ 34 ] ), FIR_Coefs[  1 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  2 ], buf_ptr[ 33 ] ), FIR_Coefs[  2 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  3 ], buf_ptr[ 32 ] ), FIR_Coefs[  3 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  4 ], buf_ptr[ 31 ] ), FIR_Coefs[  4 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  5 ], buf_ptr[ 30 ] ), FIR_Coefs[  5 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  6 ], buf_ptr[ 29 ] ), FIR_Coefs[  6 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  7 ], buf_ptr[ 28 ] ), FIR_Coefs[  7 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  8 ], buf_ptr[ 27 ] ), FIR_Coefs[  8 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[  9 ], buf_ptr[ 26 ] ), FIR_Coefs[  9 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 10 ], buf_ptr[ 25 ] ), FIR_Coefs[ 10 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 11 ], buf_ptr[ 24 ] ), FIR_Coefs[ 11 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 12 ], buf_ptr[ 23 ] ), FIR_Coefs[ 12 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 13 ], buf_ptr[ 22 ] ), FIR_Coefs[ 13 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 14 ], buf_ptr[ 21 ] ), FIR_Coefs[ 14 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 15 ], buf_ptr[ 20 ] ), FIR_Coefs[ 15 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 16 ], buf_ptr[ 19 ] ), FIR_Coefs[ 16 ] );
                res_Q6 = silk_SMLAWB( res_Q6, silk_ADD32( buf_ptr[ 17 ], buf_ptr[ 18 ] ), FIR_Coefs[ 17 ] );

                /* Scale down, saturate and store in output array */
                *out++ = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( res_Q6, 6 ) );
            }
            break;
        default:
            celt_assert( 0 );
    }
    return out;
}

/* Resample with a 2nd order AR filter followed by FIR interpolation */
void silk_resampler_private_down_FIR(
    void                            *SS,            /* I/O  Resampler state             */
    opus_int16                      out[],          /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    opus_int32                      inLen           /* I    Number of input samples     */
)
{
    silk_resampler_state_struct *S = (silk_resampler_state_struct *)SS;
    opus_int32 nSamplesIn;
    opus_int32 max_index_Q16, index_increment_Q16;
    VARDECL( opus_int32, buf );
    const opus_int16 *FIR_Coefs;
    SAVE_STACK;

    ALLOC( buf, S->batchSize + S->FIR_Order, opus_int32 );

    /* Copy buffered samples to start of buffer */
    silk_memcpy( buf, S->sFIR.i32, S->FIR_Order * sizeof( opus_int32 ) );

    FIR_Coefs = &S->Coefs[ 2 ];

    /* Iterate over blocks of frameSizeIn input samples */
    index_increment_Q16 = S->invRatio_Q16;
    while( 1 ) {
        nSamplesIn = silk_min( inLen, S->batchSize );

        /* Second-order AR filter (output in Q8) */
        silk_resampler_private_AR2( S->sIIR, &buf[ S->FIR_Order ], in, S->Coefs, nSamplesIn );

        max_index_Q16 = silk_LSHIFT32( nSamplesIn, 16 );

        /* Interpolate filtered signal */
        out = silk_resampler_private_down_FIR_INTERPOL( out, buf, FIR_Coefs, S->FIR_Order,
            S->FIR_Fracs, max_index_Q16, index_increment_Q16 );

        in += nSamplesIn;
        inLen -= nSamplesIn;

        if( inLen > 1 ) {
            /* More iterations to do; copy last part of filtered signal to beginning of buffer */
            silk_memcpy( buf, &buf[ nSamplesIn ], S->FIR_Order * sizeof( opus_int32 ) );
        } else {
            break;
        }
    }

    /* Copy last part of filtered signal to the state for the next call */
    silk_memcpy( S->sFIR.i32, &buf[ nSamplesIn ], S->FIR_Order * sizeof( opus_int32 ) );
    RESTORE_STACK;
}
