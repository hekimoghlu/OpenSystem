/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

#include "main_FIX.h"
#include "stack_alloc.h"
#include "tuning_parameters.h"

/* Find pitch lags */
void silk_find_pitch_lags_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  encoder control                                                             */
    opus_int16                      res[],                                  /* O    residual                                                                    */
    const opus_int16                x[],                                    /* I    Speech signal                                                               */
    int                             arch                                    /* I    Run-time architecture                                                       */
)
{
    opus_int   buf_len, i, scale;
    opus_int32 thrhld_Q13, res_nrg;
    const opus_int16 *x_ptr;
    VARDECL( opus_int16, Wsig );
    opus_int16 *Wsig_ptr;
    opus_int32 auto_corr[ MAX_FIND_PITCH_LPC_ORDER + 1 ];
    opus_int16 rc_Q15[    MAX_FIND_PITCH_LPC_ORDER ];
    opus_int32 A_Q24[     MAX_FIND_PITCH_LPC_ORDER ];
    opus_int16 A_Q12[     MAX_FIND_PITCH_LPC_ORDER ];
    SAVE_STACK;

    /******************************************/
    /* Set up buffer lengths etc based on Fs  */
    /******************************************/
    buf_len = psEnc->sCmn.la_pitch + psEnc->sCmn.frame_length + psEnc->sCmn.ltp_mem_length;

    /* Safety check */
    celt_assert( buf_len >= psEnc->sCmn.pitch_LPC_win_length );

    /*************************************/
    /* Estimate LPC AR coefficients      */
    /*************************************/

    /* Calculate windowed signal */

    ALLOC( Wsig, psEnc->sCmn.pitch_LPC_win_length, opus_int16 );

    /* First LA_LTP samples */
    x_ptr = x + buf_len - psEnc->sCmn.pitch_LPC_win_length;
    Wsig_ptr  = Wsig;
    silk_apply_sine_window( Wsig_ptr, x_ptr, 1, psEnc->sCmn.la_pitch );

    /* Middle un - windowed samples */
    Wsig_ptr  += psEnc->sCmn.la_pitch;
    x_ptr += psEnc->sCmn.la_pitch;
    silk_memcpy( Wsig_ptr, x_ptr, ( psEnc->sCmn.pitch_LPC_win_length - silk_LSHIFT( psEnc->sCmn.la_pitch, 1 ) ) * sizeof( opus_int16 ) );

    /* Last LA_LTP samples */
    Wsig_ptr  += psEnc->sCmn.pitch_LPC_win_length - silk_LSHIFT( psEnc->sCmn.la_pitch, 1 );
    x_ptr += psEnc->sCmn.pitch_LPC_win_length - silk_LSHIFT( psEnc->sCmn.la_pitch, 1 );
    silk_apply_sine_window( Wsig_ptr, x_ptr, 2, psEnc->sCmn.la_pitch );

    /* Calculate autocorrelation sequence */
    silk_autocorr( auto_corr, &scale, Wsig, psEnc->sCmn.pitch_LPC_win_length, psEnc->sCmn.pitchEstimationLPCOrder + 1, arch );

    /* Add white noise, as fraction of energy */
    auto_corr[ 0 ] = silk_SMLAWB( auto_corr[ 0 ], auto_corr[ 0 ], SILK_FIX_CONST( FIND_PITCH_WHITE_NOISE_FRACTION, 16 ) ) + 1;

    /* Calculate the reflection coefficients using schur */
    res_nrg = silk_schur( rc_Q15, auto_corr, psEnc->sCmn.pitchEstimationLPCOrder );

    /* Prediction gain */
    psEncCtrl->predGain_Q16 = silk_DIV32_varQ( auto_corr[ 0 ], silk_max_int( res_nrg, 1 ), 16 );

    /* Convert reflection coefficients to prediction coefficients */
    silk_k2a( A_Q24, rc_Q15, psEnc->sCmn.pitchEstimationLPCOrder );

    /* Convert From 32 bit Q24 to 16 bit Q12 coefs */
    for( i = 0; i < psEnc->sCmn.pitchEstimationLPCOrder; i++ ) {
        A_Q12[ i ] = (opus_int16)silk_SAT16( silk_RSHIFT( A_Q24[ i ], 12 ) );
    }

    /* Do BWE */
    silk_bwexpander( A_Q12, psEnc->sCmn.pitchEstimationLPCOrder, SILK_FIX_CONST( FIND_PITCH_BANDWIDTH_EXPANSION, 16 ) );

    /*****************************************/
    /* LPC analysis filtering                */
    /*****************************************/
    silk_LPC_analysis_filter( res, x, A_Q12, buf_len, psEnc->sCmn.pitchEstimationLPCOrder, psEnc->sCmn.arch );

    if( psEnc->sCmn.indices.signalType != TYPE_NO_VOICE_ACTIVITY && psEnc->sCmn.first_frame_after_reset == 0 ) {
        /* Threshold for pitch estimator */
        thrhld_Q13 = SILK_FIX_CONST( 0.6, 13 );
        thrhld_Q13 = silk_SMLABB( thrhld_Q13, SILK_FIX_CONST( -0.004, 13 ), psEnc->sCmn.pitchEstimationLPCOrder );
        thrhld_Q13 = silk_SMLAWB( thrhld_Q13, SILK_FIX_CONST( -0.1,   21  ), psEnc->sCmn.speech_activity_Q8 );
        thrhld_Q13 = silk_SMLABB( thrhld_Q13, SILK_FIX_CONST( -0.15,  13 ), silk_RSHIFT( psEnc->sCmn.prevSignalType, 1 ) );
        thrhld_Q13 = silk_SMLAWB( thrhld_Q13, SILK_FIX_CONST( -0.1,   14 ), psEnc->sCmn.input_tilt_Q15 );
        thrhld_Q13 = silk_SAT16(  thrhld_Q13 );

        /*****************************************/
        /* Call pitch estimator                  */
        /*****************************************/
        if( silk_pitch_analysis_core( res, psEncCtrl->pitchL, &psEnc->sCmn.indices.lagIndex, &psEnc->sCmn.indices.contourIndex,
                &psEnc->LTPCorr_Q15, psEnc->sCmn.prevLag, psEnc->sCmn.pitchEstimationThreshold_Q16,
                (opus_int)thrhld_Q13, psEnc->sCmn.fs_kHz, psEnc->sCmn.pitchEstimationComplexity, psEnc->sCmn.nb_subfr,
                psEnc->sCmn.arch) == 0 )
        {
            psEnc->sCmn.indices.signalType = TYPE_VOICED;
        } else {
            psEnc->sCmn.indices.signalType = TYPE_UNVOICED;
        }
    } else {
        silk_memset( psEncCtrl->pitchL, 0, sizeof( psEncCtrl->pitchL ) );
        psEnc->sCmn.indices.lagIndex = 0;
        psEnc->sCmn.indices.contourIndex = 0;
        psEnc->LTPCorr_Q15 = 0;
    }
    RESTORE_STACK;
}
