/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#ifndef SILK_STRUCTS_FLP_H
#define SILK_STRUCTS_FLP_H

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
    silk_float                  HarmShapeGain_smth;
    silk_float                  Tilt_smth;
} silk_shape_state_FLP;

/********************************/
/* Encoder state FLP            */
/********************************/
typedef struct {
    silk_encoder_state          sCmn;                               /* Common struct, shared with fixed-point code */
    silk_shape_state_FLP        sShape;                             /* Noise shaping state */

    /* Buffer for find pitch and noise shape analysis */
    silk_float                  x_buf[ 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX ];/* Buffer for find pitch and noise shape analysis */
    silk_float                  LTPCorr;                            /* Normalized correlation from pitch lag estimator */
} silk_encoder_state_FLP;

/************************/
/* Encoder control FLP  */
/************************/
typedef struct {
    /* Prediction and coding parameters */
    silk_float                  Gains[ MAX_NB_SUBFR ];
    silk_float                  PredCoef[ 2 ][ MAX_LPC_ORDER ];     /* holds interpolated and final coefficients */
    silk_float                  LTPCoef[LTP_ORDER * MAX_NB_SUBFR];
    silk_float                  LTP_scale;
    opus_int                    pitchL[ MAX_NB_SUBFR ];

    /* Noise shaping parameters */
    silk_float                  AR[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ];
    silk_float                  LF_MA_shp[     MAX_NB_SUBFR ];
    silk_float                  LF_AR_shp[     MAX_NB_SUBFR ];
    silk_float                  Tilt[          MAX_NB_SUBFR ];
    silk_float                  HarmShapeGain[ MAX_NB_SUBFR ];
    silk_float                  Lambda;
    silk_float                  input_quality;
    silk_float                  coding_quality;

    /* Measures */
    silk_float                  predGain;
    silk_float                  LTPredCodGain;
    silk_float                  ResNrg[ MAX_NB_SUBFR ];             /* Residual energy per subframe */

    /* Parameters for CBR mode */
    opus_int32                  GainsUnq_Q16[ MAX_NB_SUBFR ];
    opus_int8                   lastGainIndexPrev;
} silk_encoder_control_FLP;

/************************/
/* Encoder Super Struct */
/************************/
typedef struct {
    silk_encoder_state_FLP      state_Fxx[ ENCODER_NUM_CHANNELS ];
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
