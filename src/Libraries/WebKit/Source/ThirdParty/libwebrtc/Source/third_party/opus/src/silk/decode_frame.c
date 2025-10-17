/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "PLC.h"

/****************/
/* Decode frame */
/****************/
opus_int silk_decode_frame(
    silk_decoder_state          *psDec,                         /* I/O  Pointer to Silk decoder state               */
    ec_dec                      *psRangeDec,                    /* I/O  Compressor data structure                   */
    opus_int16                  pOut[],                         /* O    Pointer to output speech frame              */
    opus_int32                  *pN,                            /* O    Pointer to size of output frame             */
    opus_int                    lostFlag,                       /* I    0: no loss, 1 loss, 2 decode fec            */
    opus_int                    condCoding,                     /* I    The type of conditional coding to use       */
    int                         arch                            /* I    Run-time architecture                       */
)
{
    VARDECL( silk_decoder_control, psDecCtrl );
    opus_int         L, mv_len, ret = 0;
    SAVE_STACK;

    L = psDec->frame_length;
    ALLOC( psDecCtrl, 1, silk_decoder_control );
    psDecCtrl->LTP_scale_Q14 = 0;

    /* Safety checks */
    celt_assert( L > 0 && L <= MAX_FRAME_LENGTH );

    if(   lostFlag == FLAG_DECODE_NORMAL ||
        ( lostFlag == FLAG_DECODE_LBRR && psDec->LBRR_flags[ psDec->nFramesDecoded ] == 1 ) )
    {
        VARDECL( opus_int16, pulses );
        ALLOC( pulses, (L + SHELL_CODEC_FRAME_LENGTH - 1) &
                       ~(SHELL_CODEC_FRAME_LENGTH - 1), opus_int16 );
        /*********************************************/
        /* Decode quantization indices of side info  */
        /*********************************************/
        silk_decode_indices( psDec, psRangeDec, psDec->nFramesDecoded, lostFlag, condCoding );

        /*********************************************/
        /* Decode quantization indices of excitation */
        /*********************************************/
        silk_decode_pulses( psRangeDec, pulses, psDec->indices.signalType,
                psDec->indices.quantOffsetType, psDec->frame_length );

        /********************************************/
        /* Decode parameters and pulse signal       */
        /********************************************/
        silk_decode_parameters( psDec, psDecCtrl, condCoding );

        /********************************************************/
        /* Run inverse NSQ                                      */
        /********************************************************/
        silk_decode_core( psDec, psDecCtrl, pOut, pulses, arch );

        /********************************************************/
        /* Update PLC state                                     */
        /********************************************************/
        silk_PLC( psDec, psDecCtrl, pOut, 0, arch );

        psDec->lossCnt = 0;
        psDec->prevSignalType = psDec->indices.signalType;
        celt_assert( psDec->prevSignalType >= 0 && psDec->prevSignalType <= 2 );

        /* A frame has been decoded without errors */
        psDec->first_frame_after_reset = 0;
    } else {
        /* Handle packet loss by extrapolation */
        silk_PLC( psDec, psDecCtrl, pOut, 1, arch );
    }

    /*************************/
    /* Update output buffer. */
    /*************************/
    celt_assert( psDec->ltp_mem_length >= psDec->frame_length );
    mv_len = psDec->ltp_mem_length - psDec->frame_length;
    silk_memmove( psDec->outBuf, &psDec->outBuf[ psDec->frame_length ], mv_len * sizeof(opus_int16) );
    silk_memcpy( &psDec->outBuf[ mv_len ], pOut, psDec->frame_length * sizeof( opus_int16 ) );

    /************************************************/
    /* Comfort noise generation / estimation        */
    /************************************************/
    silk_CNG( psDec, psDecCtrl, pOut, L );

    /****************************************************************/
    /* Ensure smooth connection of extrapolated and good frames     */
    /****************************************************************/
    silk_PLC_glue_frames( psDec, pOut, L );

    /* Update some decoder state variables */
    psDec->lagPrev = psDecCtrl->pitchL[ psDec->nb_subfr - 1 ];

    /* Set output frame length */
    *pN = L;

    RESTORE_STACK;
    return ret;
}
