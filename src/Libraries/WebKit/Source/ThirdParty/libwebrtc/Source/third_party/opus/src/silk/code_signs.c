/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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

/*#define silk_enc_map(a)                ((a) > 0 ? 1 : 0)*/
/*#define silk_dec_map(a)                ((a) > 0 ? 1 : -1)*/
/* shifting avoids if-statement */
#define silk_enc_map(a)                  ( silk_RSHIFT( (a), 15 ) + 1 )
#define silk_dec_map(a)                  ( silk_LSHIFT( (a),  1 ) - 1 )

/* Encodes signs of excitation */
void silk_encode_signs(
    ec_enc                      *psRangeEnc,                        /* I/O  Compressor data structure                   */
    const opus_int8             pulses[],                           /* I    pulse signal                                */
    opus_int                    length,                             /* I    length of input                             */
    const opus_int              signalType,                         /* I    Signal type                                 */
    const opus_int              quantOffsetType,                    /* I    Quantization offset type                    */
    const opus_int              sum_pulses[ MAX_NB_SHELL_BLOCKS ]   /* I    Sum of absolute pulses per block            */
)
{
    opus_int         i, j, p;
    opus_uint8       icdf[ 2 ];
    const opus_int8  *q_ptr;
    const opus_uint8 *icdf_ptr;

    icdf[ 1 ] = 0;
    q_ptr = pulses;
    i = silk_SMULBB( 7, silk_ADD_LSHIFT( quantOffsetType, signalType, 1 ) );
    icdf_ptr = &silk_sign_iCDF[ i ];
    length = silk_RSHIFT( length + SHELL_CODEC_FRAME_LENGTH/2, LOG2_SHELL_CODEC_FRAME_LENGTH );
    for( i = 0; i < length; i++ ) {
        p = sum_pulses[ i ];
        if( p > 0 ) {
            icdf[ 0 ] = icdf_ptr[ silk_min( p & 0x1F, 6 ) ];
            for( j = 0; j < SHELL_CODEC_FRAME_LENGTH; j++ ) {
                if( q_ptr[ j ] != 0 ) {
                    ec_enc_icdf( psRangeEnc, silk_enc_map( q_ptr[ j ]), icdf, 8 );
                }
            }
        }
        q_ptr += SHELL_CODEC_FRAME_LENGTH;
    }
}

/* Decodes signs of excitation */
void silk_decode_signs(
    ec_dec                      *psRangeDec,                        /* I/O  Compressor data structure                   */
    opus_int16                  pulses[],                           /* I/O  pulse signal                                */
    opus_int                    length,                             /* I    length of input                             */
    const opus_int              signalType,                         /* I    Signal type                                 */
    const opus_int              quantOffsetType,                    /* I    Quantization offset type                    */
    const opus_int              sum_pulses[ MAX_NB_SHELL_BLOCKS ]   /* I    Sum of absolute pulses per block            */
)
{
    opus_int         i, j, p;
    opus_uint8       icdf[ 2 ];
    opus_int16       *q_ptr;
    const opus_uint8 *icdf_ptr;

    icdf[ 1 ] = 0;
    q_ptr = pulses;
    i = silk_SMULBB( 7, silk_ADD_LSHIFT( quantOffsetType, signalType, 1 ) );
    icdf_ptr = &silk_sign_iCDF[ i ];
    length = silk_RSHIFT( length + SHELL_CODEC_FRAME_LENGTH/2, LOG2_SHELL_CODEC_FRAME_LENGTH );
    for( i = 0; i < length; i++ ) {
        p = sum_pulses[ i ];
        if( p > 0 ) {
            icdf[ 0 ] = icdf_ptr[ silk_min( p & 0x1F, 6 ) ];
            for( j = 0; j < SHELL_CODEC_FRAME_LENGTH; j++ ) {
                if( q_ptr[ j ] > 0 ) {
                    /* attach sign */
#if 0
                    /* conditional implementation */
                    if( ec_dec_icdf( psRangeDec, icdf, 8 ) == 0 ) {
                        q_ptr[ j ] = -q_ptr[ j ];
                    }
#else
                    /* implementation with shift, subtraction, multiplication */
                    q_ptr[ j ] *= silk_dec_map( ec_dec_icdf( psRangeDec, icdf, 8 ) );
#endif
                }
            }
        }
        q_ptr += SHELL_CODEC_FRAME_LENGTH;
    }
}
