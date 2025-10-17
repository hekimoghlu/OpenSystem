/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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

#include "main_FLP.h"

/* Wrappers. Calls flp / fix code */

/* Convert AR filter coefficients to NLSF parameters */
void silk_A2NLSF_FLP(
    opus_int16                      *NLSF_Q15,                          /* O    NLSF vector      [ LPC_order ]              */
    const silk_float                *pAR,                               /* I    LPC coefficients [ LPC_order ]              */
    const opus_int                  LPC_order                           /* I    LPC order                                   */
)
{
    opus_int   i;
    opus_int32 a_fix_Q16[ MAX_LPC_ORDER ];

    for( i = 0; i < LPC_order; i++ ) {
        a_fix_Q16[ i ] = silk_float2int( pAR[ i ] * 65536.0f );
    }

    silk_A2NLSF( NLSF_Q15, a_fix_Q16, LPC_order );
}

/* Convert LSF parameters to AR prediction filter coefficients */
void silk_NLSF2A_FLP(
    silk_float                      *pAR,                               /* O    LPC coefficients [ LPC_order ]              */
    const opus_int16                *NLSF_Q15,                          /* I    NLSF vector      [ LPC_order ]              */
    const opus_int                  LPC_order,                          /* I    LPC order                                   */
    int                             arch                                /* I    Run-time architecture                       */
)
{
    opus_int   i;
    opus_int16 a_fix_Q12[ MAX_LPC_ORDER ];

    silk_NLSF2A( a_fix_Q12, NLSF_Q15, LPC_order, arch );

    for( i = 0; i < LPC_order; i++ ) {
        pAR[ i ] = ( silk_float )a_fix_Q12[ i ] * ( 1.0f / 4096.0f );
    }
}

/******************************************/
/* Floating-point NLSF processing wrapper */
/******************************************/
void silk_process_NLSFs_FLP(
    silk_encoder_state              *psEncC,                            /* I/O  Encoder state                               */
    silk_float                      PredCoef[ 2 ][ MAX_LPC_ORDER ],     /* O    Prediction coefficients                     */
    opus_int16                      NLSF_Q15[      MAX_LPC_ORDER ],     /* I/O  Normalized LSFs (quant out) (0 - (2^15-1))  */
    const opus_int16                prev_NLSF_Q15[ MAX_LPC_ORDER ]      /* I    Previous Normalized LSFs (0 - (2^15-1))     */
)
{
    opus_int     i, j;
    opus_int16   PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];

    silk_process_NLSFs( psEncC, PredCoef_Q12, NLSF_Q15, prev_NLSF_Q15);

    for( j = 0; j < 2; j++ ) {
        for( i = 0; i < psEncC->predictLPCOrder; i++ ) {
            PredCoef[ j ][ i ] = ( silk_float )PredCoef_Q12[ j ][ i ] * ( 1.0f / 4096.0f );
        }
    }
}

/****************************************/
/* Floating-point Silk NSQ wrapper      */
/****************************************/
void silk_NSQ_wrapper_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    SideInfoIndices                 *psIndices,                         /* I/O  Quantization indices                        */
    silk_nsq_state                  *psNSQ,                             /* I/O  Noise Shaping Quantzation state             */
    opus_int8                       pulses[],                           /* O    Quantized pulse signal                      */
    const silk_float                x[]                                 /* I    Prefiltered input signal                    */
)
{
    opus_int     i, j;
    opus_int16   x16[ MAX_FRAME_LENGTH ];
    opus_int32   Gains_Q16[ MAX_NB_SUBFR ];
    silk_DWORD_ALIGN opus_int16 PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    opus_int16   LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    opus_int     LTP_scale_Q14;

    /* Noise shaping parameters */
    opus_int16   AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    opus_int32   LF_shp_Q14[ MAX_NB_SUBFR ];         /* Packs two int16 coefficients per int32 value             */
    opus_int     Lambda_Q10;
    opus_int     Tilt_Q14[ MAX_NB_SUBFR ];
    opus_int     HarmShapeGain_Q14[ MAX_NB_SUBFR ];

    /* Convert control struct to fix control struct */
    /* Noise shape parameters */
    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        for( j = 0; j < psEnc->sCmn.shapingLPCOrder; j++ ) {
            AR_Q13[ i * MAX_SHAPE_LPC_ORDER + j ] = silk_float2int( psEncCtrl->AR[ i * MAX_SHAPE_LPC_ORDER + j ] * 8192.0f );
        }
    }

    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        LF_shp_Q14[ i ] =   silk_LSHIFT32( silk_float2int( psEncCtrl->LF_AR_shp[ i ]     * 16384.0f ), 16 ) |
                              (opus_uint16)silk_float2int( psEncCtrl->LF_MA_shp[ i ]     * 16384.0f );
        Tilt_Q14[ i ]   =        (opus_int)silk_float2int( psEncCtrl->Tilt[ i ]          * 16384.0f );
        HarmShapeGain_Q14[ i ] = (opus_int)silk_float2int( psEncCtrl->HarmShapeGain[ i ] * 16384.0f );
    }
    Lambda_Q10 = ( opus_int )silk_float2int( psEncCtrl->Lambda * 1024.0f );

    /* prediction and coding parameters */
    for( i = 0; i < psEnc->sCmn.nb_subfr * LTP_ORDER; i++ ) {
        LTPCoef_Q14[ i ] = (opus_int16)silk_float2int( psEncCtrl->LTPCoef[ i ] * 16384.0f );
    }

    for( j = 0; j < 2; j++ ) {
        for( i = 0; i < psEnc->sCmn.predictLPCOrder; i++ ) {
            PredCoef_Q12[ j ][ i ] = (opus_int16)silk_float2int( psEncCtrl->PredCoef[ j ][ i ] * 4096.0f );
        }
    }

    for( i = 0; i < psEnc->sCmn.nb_subfr; i++ ) {
        Gains_Q16[ i ] = silk_float2int( psEncCtrl->Gains[ i ] * 65536.0f );
        silk_assert( Gains_Q16[ i ] > 0 );
    }

    if( psIndices->signalType == TYPE_VOICED ) {
        LTP_scale_Q14 = silk_LTPScales_table_Q14[ psIndices->LTP_scaleIndex ];
    } else {
        LTP_scale_Q14 = 0;
    }

    /* Convert input to fix */
    for( i = 0; i < psEnc->sCmn.frame_length; i++ ) {
        x16[ i ] = silk_float2int( x[ i ] );
    }

    /* Call NSQ */
    if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
        silk_NSQ_del_dec( &psEnc->sCmn, psNSQ, psIndices, x16, pulses, PredCoef_Q12[ 0 ], LTPCoef_Q14,
            AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, psEncCtrl->pitchL, Lambda_Q10, LTP_scale_Q14, psEnc->sCmn.arch );
    } else {
        silk_NSQ( &psEnc->sCmn, psNSQ, psIndices, x16, pulses, PredCoef_Q12[ 0 ], LTPCoef_Q14,
            AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, psEncCtrl->pitchL, Lambda_Q10, LTP_scale_Q14, psEnc->sCmn.arch );
    }
}

/***********************************************/
/* Floating-point Silk LTP quantiation wrapper */
/***********************************************/
void silk_quant_LTP_gains_FLP(
    silk_float                      B[ MAX_NB_SUBFR * LTP_ORDER ],      /* O    Quantized LTP gains                            */
    opus_int8                       cbk_index[ MAX_NB_SUBFR ],          /* O    Codebook index                              */
    opus_int8                       *periodicity_index,                 /* O    Periodicity index                           */
    opus_int32                      *sum_log_gain_Q7,                   /* I/O  Cumulative max prediction gain  */
    silk_float                      *pred_gain_dB,                        /* O    LTP prediction gain                            */
    const silk_float                XX[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ], /* I    Correlation matrix                    */
    const silk_float                xX[ MAX_NB_SUBFR * LTP_ORDER ],        /* I    Correlation vector                            */
    const opus_int                    subfr_len,                            /* I    Number of samples per subframe                */
    const opus_int                    nb_subfr,                           /* I    Number of subframes                            */
    int                             arch                                /* I    Run-time architecture                       */
)
{
    opus_int   i, pred_gain_dB_Q7;
    opus_int16 B_Q14[ MAX_NB_SUBFR * LTP_ORDER ];
    opus_int32 XX_Q17[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ];
    opus_int32 xX_Q17[ MAX_NB_SUBFR * LTP_ORDER ];

    i = 0;
    do {
        XX_Q17[ i ] = (opus_int32)silk_float2int( XX[ i ] * 131072.0f );
    } while ( ++i < nb_subfr * LTP_ORDER * LTP_ORDER );
    i = 0;
    do {
        xX_Q17[ i ] = (opus_int32)silk_float2int( xX[ i ] * 131072.0f );
    } while ( ++i < nb_subfr * LTP_ORDER );

    silk_quant_LTP_gains( B_Q14, cbk_index, periodicity_index, sum_log_gain_Q7, &pred_gain_dB_Q7, XX_Q17, xX_Q17, subfr_len, nb_subfr, arch );

    for( i = 0; i < nb_subfr * LTP_ORDER; i++ ) {
        B[ i ] = (silk_float)B_Q14[ i ] * ( 1.0f / 16384.0f );
    }

    *pred_gain_dB = (silk_float)pred_gain_dB_Q7 * ( 1.0f / 128.0f );
}
