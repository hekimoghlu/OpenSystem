/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

/***********************************************************
* Pitch analyser function
********************************************************** */
#include "SigProc_FIX.h"
#include "pitch_est_defines.h"

void silk_decode_pitch(
    opus_int16                  lagIndex,           /* I                                                                */
    opus_int8                   contourIndex,       /* O                                                                */
    opus_int                    pitch_lags[],       /* O    4 pitch values                                              */
    const opus_int              Fs_kHz,             /* I    sampling frequency (kHz)                                    */
    const opus_int              nb_subfr            /* I    number of sub frames                                        */
)
{
    opus_int   lag, k, min_lag, max_lag, cbk_size;
    const opus_int8 *Lag_CB_ptr;

    if( Fs_kHz == 8 ) {
        if( nb_subfr == PE_MAX_NB_SUBFR ) {
            Lag_CB_ptr = &silk_CB_lags_stage2[ 0 ][ 0 ];
            cbk_size   = PE_NB_CBKS_STAGE2_EXT;
        } else {
            celt_assert( nb_subfr == PE_MAX_NB_SUBFR >> 1 );
            Lag_CB_ptr = &silk_CB_lags_stage2_10_ms[ 0 ][ 0 ];
            cbk_size   = PE_NB_CBKS_STAGE2_10MS;
        }
    } else {
        if( nb_subfr == PE_MAX_NB_SUBFR ) {
            Lag_CB_ptr = &silk_CB_lags_stage3[ 0 ][ 0 ];
            cbk_size   = PE_NB_CBKS_STAGE3_MAX;
        } else {
            celt_assert( nb_subfr == PE_MAX_NB_SUBFR >> 1 );
            Lag_CB_ptr = &silk_CB_lags_stage3_10_ms[ 0 ][ 0 ];
            cbk_size   = PE_NB_CBKS_STAGE3_10MS;
        }
    }

    min_lag = silk_SMULBB( PE_MIN_LAG_MS, Fs_kHz );
    max_lag = silk_SMULBB( PE_MAX_LAG_MS, Fs_kHz );
    lag = min_lag + lagIndex;

    for( k = 0; k < nb_subfr; k++ ) {
        pitch_lags[ k ] = lag + matrix_ptr( Lag_CB_ptr, k, contourIndex, cbk_size );
        pitch_lags[ k ] = silk_LIMIT( pitch_lags[ k ], min_lag, max_lag );
    }
}
