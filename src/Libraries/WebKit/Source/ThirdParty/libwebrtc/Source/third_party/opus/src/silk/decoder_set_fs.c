/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

/* Set decoder sampling rate */
opus_int silk_decoder_set_fs(
    silk_decoder_state          *psDec,                         /* I/O  Decoder state pointer                       */
    opus_int                    fs_kHz,                         /* I    Sampling frequency (kHz)                    */
    opus_int32                  fs_API_Hz                       /* I    API Sampling frequency (Hz)                 */
)
{
    opus_int frame_length, ret = 0;

    celt_assert( fs_kHz == 8 || fs_kHz == 12 || fs_kHz == 16 );
    celt_assert( psDec->nb_subfr == MAX_NB_SUBFR || psDec->nb_subfr == MAX_NB_SUBFR/2 );

    /* New (sub)frame length */
    psDec->subfr_length = silk_SMULBB( SUB_FRAME_LENGTH_MS, fs_kHz );
    frame_length = silk_SMULBB( psDec->nb_subfr, psDec->subfr_length );

    /* Initialize resampler when switching internal or external sampling frequency */
    if( psDec->fs_kHz != fs_kHz || psDec->fs_API_hz != fs_API_Hz ) {
        /* Initialize the resampler for dec_API.c preparing resampling from fs_kHz to API_fs_Hz */
        ret += silk_resampler_init( &psDec->resampler_state, silk_SMULBB( fs_kHz, 1000 ), fs_API_Hz, 0 );

        psDec->fs_API_hz = fs_API_Hz;
    }

    if( psDec->fs_kHz != fs_kHz || frame_length != psDec->frame_length ) {
        if( fs_kHz == 8 ) {
            if( psDec->nb_subfr == MAX_NB_SUBFR ) {
                psDec->pitch_contour_iCDF = silk_pitch_contour_NB_iCDF;
            } else {
                psDec->pitch_contour_iCDF = silk_pitch_contour_10_ms_NB_iCDF;
            }
        } else {
            if( psDec->nb_subfr == MAX_NB_SUBFR ) {
                psDec->pitch_contour_iCDF = silk_pitch_contour_iCDF;
            } else {
                psDec->pitch_contour_iCDF = silk_pitch_contour_10_ms_iCDF;
            }
        }
        if( psDec->fs_kHz != fs_kHz ) {
            psDec->ltp_mem_length = silk_SMULBB( LTP_MEM_LENGTH_MS, fs_kHz );
            if( fs_kHz == 8 || fs_kHz == 12 ) {
                psDec->LPC_order = MIN_LPC_ORDER;
                psDec->psNLSF_CB = &silk_NLSF_CB_NB_MB;
            } else {
                psDec->LPC_order = MAX_LPC_ORDER;
                psDec->psNLSF_CB = &silk_NLSF_CB_WB;
            }
            if( fs_kHz == 16 ) {
                psDec->pitch_lag_low_bits_iCDF = silk_uniform8_iCDF;
            } else if( fs_kHz == 12 ) {
                psDec->pitch_lag_low_bits_iCDF = silk_uniform6_iCDF;
            } else if( fs_kHz == 8 ) {
                psDec->pitch_lag_low_bits_iCDF = silk_uniform4_iCDF;
            } else {
                /* unsupported sampling rate */
                celt_assert( 0 );
            }
            psDec->first_frame_after_reset = 1;
            psDec->lagPrev                 = 100;
            psDec->LastGainIndex           = 10;
            psDec->prevSignalType          = TYPE_NO_VOICE_ACTIVITY;
            silk_memset( psDec->outBuf, 0, sizeof(psDec->outBuf));
            silk_memset( psDec->sLPC_Q14_buf, 0, sizeof(psDec->sLPC_Q14_buf) );
        }

        psDec->fs_kHz       = fs_kHz;
        psDec->frame_length = frame_length;
    }

    /* Check that settings are valid */
    celt_assert( psDec->frame_length > 0 && psDec->frame_length <= MAX_FRAME_LENGTH );

    return ret;
}

