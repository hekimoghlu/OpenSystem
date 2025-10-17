/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#ifndef SILK_TUNING_PARAMETERS_H
#define SILK_TUNING_PARAMETERS_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Decay time for bitreservoir */
#define BITRESERVOIR_DECAY_TIME_MS                      500

/*******************/
/* Pitch estimator */
/*******************/

/* Level of noise floor for whitening filter LPC analysis in pitch analysis */
#define FIND_PITCH_WHITE_NOISE_FRACTION                 1e-3f

/* Bandwidth expansion for whitening filter in pitch analysis */
#define FIND_PITCH_BANDWIDTH_EXPANSION                  0.99f

/*********************/
/* Linear prediction */
/*********************/

/* LPC analysis regularization */
#define FIND_LPC_COND_FAC                               1e-5f

/* Max cumulative LTP gain */
#define MAX_SUM_LOG_GAIN_DB                             250.0f

/* LTP analysis defines */
#define LTP_CORR_INV_MAX                                0.03f

/***********************/
/* High pass filtering */
/***********************/

/* Smoothing parameters for low end of pitch frequency range estimation */
#define VARIABLE_HP_SMTH_COEF1                          0.1f
#define VARIABLE_HP_SMTH_COEF2                          0.015f
#define VARIABLE_HP_MAX_DELTA_FREQ                      0.4f

/* Min and max cut-off frequency values (-3 dB points) */
#define VARIABLE_HP_MIN_CUTOFF_HZ                       60
#define VARIABLE_HP_MAX_CUTOFF_HZ                       100

/***********/
/* Various */
/***********/

/* VAD threshold */
#define SPEECH_ACTIVITY_DTX_THRES                       0.05f

/* Speech Activity LBRR enable threshold */
#define LBRR_SPEECH_ACTIVITY_THRES                      0.3f

/*************************/
/* Perceptual parameters */
/*************************/

/* reduction in coding SNR during low speech activity */
#define BG_SNR_DECR_dB                                  2.0f

/* factor for reducing quantization noise during voiced speech */
#define HARM_SNR_INCR_dB                                2.0f

/* factor for reducing quantization noise for unvoiced sparse signals */
#define SPARSE_SNR_INCR_dB                              2.0f

/* threshold for sparseness measure above which to use lower quantization offset during unvoiced */
#define ENERGY_VARIATION_THRESHOLD_QNT_OFFSET           0.6f

/* warping control */
#define WARPING_MULTIPLIER                              0.015f

/* fraction added to first autocorrelation value */
#define SHAPE_WHITE_NOISE_FRACTION                      3e-5f

/* noise shaping filter chirp factor */
#define BANDWIDTH_EXPANSION                             0.94f

/* harmonic noise shaping */
#define HARMONIC_SHAPING                                0.3f

/* extra harmonic noise shaping for high bitrates or noisy input */
#define HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING       0.2f

/* parameter for shaping noise towards higher frequencies */
#define HP_NOISE_COEF                                   0.25f

/* parameter for shaping noise even more towards higher frequencies during voiced speech */
#define HARM_HP_NOISE_COEF                              0.35f

/* parameter for applying a high-pass tilt to the input signal */
#define INPUT_TILT                                      0.05f

/* parameter for extra high-pass tilt to the input signal at high rates */
#define HIGH_RATE_INPUT_TILT                            0.1f

/* parameter for reducing noise at the very low frequencies */
#define LOW_FREQ_SHAPING                                4.0f

/* less reduction of noise at the very low frequencies for signals with low SNR at low frequencies */
#define LOW_QUALITY_LOW_FREQ_SHAPING_DECR               0.5f

/* subframe smoothing coefficient for HarmBoost, HarmShapeGain, Tilt (lower -> more smoothing) */
#define SUBFR_SMTH_COEF                                 0.4f

/* parameters defining the R/D tradeoff in the residual quantizer */
#define LAMBDA_OFFSET                                   1.2f
#define LAMBDA_SPEECH_ACT                               -0.2f
#define LAMBDA_DELAYED_DECISIONS                        -0.05f
#define LAMBDA_INPUT_QUALITY                            -0.1f
#define LAMBDA_CODING_QUALITY                           -0.2f
#define LAMBDA_QUANT_OFFSET                             0.8f

/* Compensation in bitrate calculations for 10 ms modes */
#define REDUCE_BITRATE_10_MS_BPS                        2200

/* Maximum time before allowing a bandwidth transition */
#define MAX_BANDWIDTH_SWITCH_DELAY_MS                   5000

#ifdef __cplusplus
}
#endif

#endif /* SILK_TUNING_PARAMETERS_H */
