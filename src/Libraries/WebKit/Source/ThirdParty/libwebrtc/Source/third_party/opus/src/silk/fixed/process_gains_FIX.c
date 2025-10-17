/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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
#include "tuning_parameters.h"

/* Processing of gains */
void silk_process_gains_FIX(
    silk_encoder_state_FIX          *psEnc,                                 /* I/O  Encoder state                                                               */
    silk_encoder_control_FIX        *psEncCtrl,                             /* I/O  Encoder control                                                             */
    opus_int                        condCoding                              /* I    The type of conditional coding to use                                       */
)
{
    silk_shape_state_FIX *psShapeSt = &psEnc->sShape;
    opus_int     k;
    opus_int32   s_Q16, InvMaxSqrVal_Q16, gain, gain_squared, ResNrg, ResNrgPart, quant_offset_Q10;

    /* Gain reduction when LTP coding gain is high */
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        /*s = -0.5f * silk_sigmoid( 0.25f * ( psEncCtrl->LTPredCodGain - 12.0f ) ); */
        s_Q16 = -silk_sigm_Q15( silk_RSHIFT_ROUND( psEncCtrl->LTPredCodGain_Q7 - SILK_FIX_CONST( 12.0, 7 ), 4 ) );
        for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
            psEncCtrl->Gains_Q16[ k ] = silk_SMLAWB( psEncCtrl->Gains_Q16[ k ], psEncCtrl->Gains_Q16[ k ], s_Q16 );
        }
    }

    /* Limit the quantized signal */
    /* InvMaxSqrVal = pow( 2.0f, 0.33f * ( 21.0f - SNR_dB ) ) / subfr_length; */
    InvMaxSqrVal_Q16 = silk_DIV32_16( silk_log2lin(
        silk_SMULWB( SILK_FIX_CONST( 21 + 16 / 0.33, 7 ) - psEnc->sCmn.SNR_dB_Q7, SILK_FIX_CONST( 0.33, 16 ) ) ), psEnc->sCmn.subfr_length );

    for( k = 0; k < psEnc->sCmn.nb_subfr; k++ ) {
        /* Soft limit on ratio residual energy and squared gains */
        ResNrg     = psEncCtrl->ResNrg[ k ];
        ResNrgPart = silk_SMULWW( ResNrg, InvMaxSqrVal_Q16 );
        if( psEncCtrl->ResNrgQ[ k ] > 0 ) {
            ResNrgPart = silk_RSHIFT_ROUND( ResNrgPart, psEncCtrl->ResNrgQ[ k ] );
        } else {
            if( ResNrgPart >= silk_RSHIFT( silk_int32_MAX, -psEncCtrl->ResNrgQ[ k ] ) ) {
                ResNrgPart = silk_int32_MAX;
            } else {
                ResNrgPart = silk_LSHIFT( ResNrgPart, -psEncCtrl->ResNrgQ[ k ] );
            }
        }
        gain = psEncCtrl->Gains_Q16[ k ];
        gain_squared = silk_ADD_SAT32( ResNrgPart, silk_SMMUL( gain, gain ) );
        if( gain_squared < silk_int16_MAX ) {
            /* recalculate with higher precision */
            gain_squared = silk_SMLAWW( silk_LSHIFT( ResNrgPart, 16 ), gain, gain );
            silk_assert( gain_squared > 0 );
            gain = silk_SQRT_APPROX( gain_squared );                    /* Q8   */
            gain = silk_min( gain, silk_int32_MAX >> 8 );
            psEncCtrl->Gains_Q16[ k ] = silk_LSHIFT_SAT32( gain, 8 );   /* Q16  */
        } else {
            gain = silk_SQRT_APPROX( gain_squared );                    /* Q0   */
            gain = silk_min( gain, silk_int32_MAX >> 16 );
            psEncCtrl->Gains_Q16[ k ] = silk_LSHIFT_SAT32( gain, 16 );  /* Q16  */
        }
    }

    /* Save unquantized gains and gain Index */
    silk_memcpy( psEncCtrl->GainsUnq_Q16, psEncCtrl->Gains_Q16, psEnc->sCmn.nb_subfr * sizeof( opus_int32 ) );
    psEncCtrl->lastGainIndexPrev = psShapeSt->LastGainIndex;

    /* Quantize gains */
    silk_gains_quant( psEnc->sCmn.indices.GainsIndices, psEncCtrl->Gains_Q16,
        &psShapeSt->LastGainIndex, condCoding == CODE_CONDITIONALLY, psEnc->sCmn.nb_subfr );

    /* Set quantizer offset for voiced signals. Larger offset when LTP coding gain is low or tilt is high (ie low-pass) */
    if( psEnc->sCmn.indices.signalType == TYPE_VOICED ) {
        if( psEncCtrl->LTPredCodGain_Q7 + silk_RSHIFT( psEnc->sCmn.input_tilt_Q15, 8 ) > SILK_FIX_CONST( 1.0, 7 ) ) {
            psEnc->sCmn.indices.quantOffsetType = 0;
        } else {
            psEnc->sCmn.indices.quantOffsetType = 1;
        }
    }

    /* Quantizer boundary adjustment */
    quant_offset_Q10 = silk_Quantization_Offsets_Q10[ psEnc->sCmn.indices.signalType >> 1 ][ psEnc->sCmn.indices.quantOffsetType ];
    psEncCtrl->Lambda_Q10 = SILK_FIX_CONST( LAMBDA_OFFSET, 10 )
                          + silk_SMULBB( SILK_FIX_CONST( LAMBDA_DELAYED_DECISIONS, 10 ), psEnc->sCmn.nStatesDelayedDecision )
                          + silk_SMULWB( SILK_FIX_CONST( LAMBDA_SPEECH_ACT,        18 ), psEnc->sCmn.speech_activity_Q8     )
                          + silk_SMULWB( SILK_FIX_CONST( LAMBDA_INPUT_QUALITY,     12 ), psEncCtrl->input_quality_Q14       )
                          + silk_SMULWB( SILK_FIX_CONST( LAMBDA_CODING_QUALITY,    12 ), psEncCtrl->coding_quality_Q14      )
                          + silk_SMULWB( SILK_FIX_CONST( LAMBDA_QUANT_OFFSET,      16 ), quant_offset_Q10                   );

    silk_assert( psEncCtrl->Lambda_Q10 > 0 );
    silk_assert( psEncCtrl->Lambda_Q10 < SILK_FIX_CONST( 2, 10 ) );
}
