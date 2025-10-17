/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC_LEGACY_ANALOG_AGC_H_
#define MODULES_AUDIO_PROCESSING_AGC_LEGACY_ANALOG_AGC_H_

#include "modules/audio_processing/agc/legacy/digital_agc.h"
#include "modules/audio_processing/agc/legacy/gain_control.h"

namespace webrtc {

/* Analog Automatic Gain Control variables:
 * Constant declarations (inner limits inside which no changes are done)
 * In the beginning the range is narrower to widen as soon as the measure
 * 'Rxx160_LP' is inside it. Currently the starting limits are -22.2+/-1dBm0
 * and the final limits -22.2+/-2.5dBm0. These levels makes the speech signal
 * go towards -25.4dBm0 (-31.4dBov). Tuned with wbfile-31.4dBov.pcm
 * The limits are created by running the AGC with a file having the desired
 * signal level and thereafter plotting Rxx160_LP in the dBm0-domain defined
 * by out=10*log10(in/260537279.7); Set the target level to the average level
 * of our measure Rxx160_LP. Remember that the levels are in blocks of 16 in
 * Q(-7). (Example matlab code: round(db2pow(-21.2)*16/2^7) )
 */
constexpr int16_t kRxxBufferLen = 10;

static const int16_t kMsecSpeechInner = 520;
static const int16_t kMsecSpeechOuter = 340;

static const int16_t kNormalVadThreshold = 400;

static const int16_t kAlphaShortTerm = 6;  // 1 >> 6 = 0.0156
static const int16_t kAlphaLongTerm = 10;  // 1 >> 10 = 0.000977

typedef struct {
  // Configurable parameters/variables
  uint32_t fs;                // Sampling frequency
  int16_t compressionGaindB;  // Fixed gain level in dB
  int16_t targetLevelDbfs;    // Target level in -dBfs of envelope (default -3)
  int16_t agcMode;            // Hard coded mode (adaptAna/adaptDig/fixedDig)
  uint8_t limiterEnable;      // Enabling limiter (on/off (default off))
  WebRtcAgcConfig defaultConfig;
  WebRtcAgcConfig usedConfig;

  // General variables
  int16_t initFlag;
  int16_t lastError;

  // Target level parameters
  // Based on the above: analogTargetLevel = round((32767*10^(-22/20))^2*16/2^7)
  int32_t analogTargetLevel;    // = kRxxBufferLen * 846805;       -22 dBfs
  int32_t startUpperLimit;      // = kRxxBufferLen * 1066064;      -21 dBfs
  int32_t startLowerLimit;      // = kRxxBufferLen * 672641;       -23 dBfs
  int32_t upperPrimaryLimit;    // = kRxxBufferLen * 1342095;      -20 dBfs
  int32_t lowerPrimaryLimit;    // = kRxxBufferLen * 534298;       -24 dBfs
  int32_t upperSecondaryLimit;  // = kRxxBufferLen * 2677832;      -17 dBfs
  int32_t lowerSecondaryLimit;  // = kRxxBufferLen * 267783;       -27 dBfs
  uint16_t targetIdx;           // Table index for corresponding target level
  int16_t analogTarget;         // Digital reference level in ENV scale

  // Analog AGC specific variables
  int32_t filterState[8];  // For downsampling wb to nb
  int32_t upperLimit;      // Upper limit for mic energy
  int32_t lowerLimit;      // Lower limit for mic energy
  int32_t Rxx160w32;       // Average energy for one frame
  int32_t Rxx16_LPw32;     // Low pass filtered subframe energies
  int32_t Rxx160_LPw32;    // Low pass filtered frame energies
  int32_t Rxx16_LPw32Max;  // Keeps track of largest energy subframe
  int32_t Rxx16_vectorw32[kRxxBufferLen];  // Array with subframe energies
  int32_t Rxx16w32_array[2][5];            // Energy values of microphone signal
  int32_t env[2][10];                      // Envelope values of subframes

  int16_t Rxx16pos;          // Current position in the Rxx16_vectorw32
  int16_t envSum;            // Filtered scaled envelope in subframes
  int16_t vadThreshold;      // Threshold for VAD decision
  int16_t inActive;          // Inactive time in milliseconds
  int16_t msTooLow;          // Milliseconds of speech at a too low level
  int16_t msTooHigh;         // Milliseconds of speech at a too high level
  int16_t changeToSlowMode;  // Change to slow mode after some time at target
  int16_t firstCall;         // First call to the process-function
  int16_t msZero;            // Milliseconds of zero input
  int16_t msecSpeechOuterChange;  // Min ms of speech between volume changes
  int16_t msecSpeechInnerChange;  // Min ms of speech between volume changes
  int16_t activeSpeech;           // Milliseconds of active speech
  int16_t muteGuardMs;            // Counter to prevent mute action
  int16_t inQueue;                // 10 ms batch indicator

  // Microphone level variables
  int32_t micRef;         // Remember ref. mic level for virtual mic
  uint16_t gainTableIdx;  // Current position in virtual gain table
  int32_t micGainIdx;     // Gain index of mic level to increase slowly
  int32_t micVol;         // Remember volume between frames
  int32_t maxLevel;       // Max possible vol level, incl dig gain
  int32_t maxAnalog;      // Maximum possible analog volume level
  int32_t maxInit;        // Initial value of "max"
  int32_t minLevel;       // Minimum possible volume level
  int32_t minOutput;      // Minimum output volume level
  int32_t zeroCtrlMax;    // Remember max gain => don't amp low input
  int32_t lastInMicLevel;

  int16_t scale;  // Scale factor for internal volume levels
  // Structs for VAD and digital_agc
  AgcVad vadMic;
  DigitalAgc digitalAgc;

  int16_t lowLevelSignal;
} LegacyAgc;

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC_LEGACY_ANALOG_AGC_H_
