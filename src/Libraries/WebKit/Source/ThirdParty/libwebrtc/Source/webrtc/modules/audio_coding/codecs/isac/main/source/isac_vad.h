/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_ISAC_VAD_H_
#define MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_ISAC_VAD_H_

#include <stddef.h>

#include "modules/audio_coding/codecs/isac/main/source/structs.h"

void WebRtcIsac_InitPitchFilter(PitchFiltstr* pitchfiltdata);
void WebRtcIsac_InitPitchAnalysis(PitchAnalysisStruct* state);
void WebRtcIsac_InitPreFilterbank(PreFiltBankstr* prefiltdata);

double WebRtcIsac_LevDurb(double* a, double* k, double* r, size_t order);

/* The number of all-pass filter factors in an upper or lower channel*/
#define NUMBEROFCHANNELAPSECTIONS 2

/* The upper channel all-pass filter factors */
extern const float WebRtcIsac_kUpperApFactorsFloat[2];

/* The lower channel all-pass filter factors */
extern const float WebRtcIsac_kLowerApFactorsFloat[2];

void WebRtcIsac_AllPassFilter2Float(float* InOut,
                                    const float* APSectionFactors,
                                    int lengthInOut,
                                    int NumberOfSections,
                                    float* FilterState);
void WebRtcIsac_SplitAndFilterFloat(float* in,
                                    float* LP,
                                    float* HP,
                                    double* LP_la,
                                    double* HP_la,
                                    PreFiltBankstr* prefiltdata);

#endif  // MODULES_AUDIO_CODING_CODECS_ISAC_MAIN_SOURCE_ISAC_VAD_H_
