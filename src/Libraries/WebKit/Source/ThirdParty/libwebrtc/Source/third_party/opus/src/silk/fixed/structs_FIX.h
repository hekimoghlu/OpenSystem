/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#ifndef SILK_STRUCTS_FIX_H
#define SILK_STRUCTS_FIX_H

#include "typedef.h"
#include "main.h"
#include "structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

/********************************/
/* Noise shaping analysis state */
/********************************/
typedef struct {
    opus_int8                   LastGainIndex;
    opus_int32                  HarmBoost_smth_Q16;
    opus_int32                  HarmShapeGain_smth_Q16;
    opus_int32                  Tilt_smth_Q16;
} silk_shape_state_FIX;

/********************************/
/* Encoder state FIX            */
/********************************/
typedef struct {
    silk_encoder_state          sCmn;                                   /* Common struct, shared with floating-point code       */
    silk_shape_state_FIX        sShape;                                 /* Shape state                                          */

    /* Buffer for find pitch and noise shape analysis */
    silk_DWORD_ALIGN opus_int16 x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];/* Buffer for find pitch and noise shape analysis  */
    opus_int                    LTPCorr_Q15;                            /* Normalized correlation from pitch lag estimator      */
    opus_int32                    resNrgSmth;
} silk_encoder_state_FIX;

/************************/
/* Encoder control FIX  */
/************************/
typedef struct {
    /* Prediction and coding parameters */
    opus_int32                  Gains_Q16[ MAX_NB_SUBFR ];
    silk_DWORD_ALIGN opus_int16 PredCoef_Q12[ 2 ][ MAX_LPC_ORDER ];
    opus_int16                  LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ];
    opus_int                    LTP_scale_Q14;
    opus_int                    pitchL[ MAX_NB_SUBFR ];

    /* Noise shaping parameters */
    /* Testing */
    silk_DWORD_ALIGN opus_int16 AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    opus_int32                  LF_shp_Q14[        MAX_NB_SUBFR ];      /* Packs two int16 coefficients per int32 value         */
    opus_int                    Tilt_Q14[          MAX_NB_SUBFR ];
    opus_int                    HarmShapeGain_Q14[ MAX_NB_SUBFR ];
    opus_int                    Lambda_Q10;
    opus_int                    input_quality_Q14;
    opus_int                    coding_quality_Q14;

    /* measures */
    opus_int32                  predGain_Q16;
    opus_int                    LTPredCodGain_Q7;
    opus_int32                  ResNrg[ MAX_NB_SUBFR ];                 /* Residual energy per subframe                         */
    opus_int                    ResNrgQ[ MAX_NB_SUBFR ];                /* Q domain for the residual energy > 0                 */

    /* Parameters for CBR mode */
    opus_int32                  GainsUnq_Q16[ MAX_NB_SUBFR ];
    opus_int8                   lastGainIndexPrev;
} silk_encoder_control_FIX;

/************************/
/* Encoder Super Struct */
/************************/
typedef struct {
    silk_encoder_state_FIX      state_Fxx[ ENCODER_NUM_CHANNELS ];
    stereo_enc_state            sStereo;
    opus_int32                  nBitsUsedLBRR;
    opus_int32                  nBitsExceeded;
    opus_int                    nChannelsAPI;
    opus_int                    nChannelsInternal;
    opus_int                    nPrevChannelsInternal;
    opus_int                    timeSinceSwitchAllowed_ms;
    opus_int                    allowBandwidthSwitch;
    opus_int                    prev_decode_only_middle;
} silk_encoder;


#ifdef __cplusplus
}
#endif

#endif
