/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
// Gaussian probability calculations internally used in vad_core.c.

#ifndef COMMON_AUDIO_VAD_VAD_GMM_H_
#define COMMON_AUDIO_VAD_VAD_GMM_H_

#include <stdint.h>

// Calculates the probability for `input`, given that `input` comes from a
// normal distribution with mean and standard deviation (`mean`, `std`).
//
// Inputs:
//      - input         : input sample in Q4.
//      - mean          : mean input in the statistical model, Q7.
//      - std           : standard deviation, Q7.
//
// Output:
//
//      - delta         : input used when updating the model, Q11.
//                        `delta` = (`input` - `mean`) / `std`^2.
//
// Return:
//   (probability for `input`) =
//    1 / `std` * exp(-(`input` - `mean`)^2 / (2 * `std`^2));
int32_t WebRtcVad_GaussianProbability(int16_t input,
                                      int16_t mean,
                                      int16_t std,
                                      int16_t* delta);

#endif  // COMMON_AUDIO_VAD_VAD_GMM_H_
