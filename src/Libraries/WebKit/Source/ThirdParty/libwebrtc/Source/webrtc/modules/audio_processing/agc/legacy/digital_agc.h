/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC_LEGACY_DIGITAL_AGC_H_
#define MODULES_AUDIO_PROCESSING_AGC_LEGACY_DIGITAL_AGC_H_

#include "common_audio/signal_processing/include/signal_processing_library.h"

namespace webrtc {

typedef struct {
  int32_t downState[8];
  int16_t HPstate;
  int16_t counter;
  int16_t logRatio;           // log( P(active) / P(inactive) ) (Q10)
  int16_t meanLongTerm;       // Q10
  int32_t varianceLongTerm;   // Q8
  int16_t stdLongTerm;        // Q10
  int16_t meanShortTerm;      // Q10
  int32_t varianceShortTerm;  // Q8
  int16_t stdShortTerm;       // Q10
} AgcVad;                     // total = 54 bytes

typedef struct {
  int32_t capacitorSlow;
  int32_t capacitorFast;
  int32_t gain;
  int32_t gainTable[32];
  int16_t gatePrevious;
  int16_t agcMode;
  AgcVad vadNearend;
  AgcVad vadFarend;
} DigitalAgc;

int32_t WebRtcAgc_InitDigital(DigitalAgc* digitalAgcInst, int16_t agcMode);

int32_t WebRtcAgc_ComputeDigitalGains(DigitalAgc* digitalAgcInst,
                                      const int16_t* const* inNear,
                                      size_t num_bands,
                                      uint32_t FS,
                                      int16_t lowLevelSignal,
                                      int32_t gains[11]);

int32_t WebRtcAgc_ApplyDigitalGains(const int32_t gains[11],
                                    size_t num_bands,
                                    uint32_t FS,
                                    const int16_t* const* in_near,
                                    int16_t* const* out);

int32_t WebRtcAgc_AddFarendToDigital(DigitalAgc* digitalAgcInst,
                                     const int16_t* inFar,
                                     size_t nrSamples);

void WebRtcAgc_InitVad(AgcVad* vadInst);

int16_t WebRtcAgc_ProcessVad(AgcVad* vadInst,    // (i) VAD state
                             const int16_t* in,  // (i) Speech signal
                             size_t nrSamples);  // (i) number of samples

int32_t WebRtcAgc_CalculateGainTable(int32_t* gainTable,         // Q16
                                     int16_t compressionGaindB,  // Q0 (in dB)
                                     int16_t targetLevelDbfs,    // Q0 (in dB)
                                     uint8_t limiterEnable,
                                     int16_t analogTarget);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC_LEGACY_DIGITAL_AGC_H_
