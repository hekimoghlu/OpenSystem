/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
// Header file including the delay estimator handle used for testing.

#ifndef MODULES_AUDIO_PROCESSING_UTILITY_DELAY_ESTIMATOR_INTERNAL_H_
#define MODULES_AUDIO_PROCESSING_UTILITY_DELAY_ESTIMATOR_INTERNAL_H_

#include "modules/audio_processing/utility/delay_estimator.h"

namespace webrtc {

typedef union {
  float float_;
  int32_t int32_;
} SpectrumType;

typedef struct {
  // Pointers to mean values of spectrum.
  SpectrumType* mean_far_spectrum;
  // `mean_far_spectrum` initialization indicator.
  int far_spectrum_initialized;

  int spectrum_size;

  // Far-end part of binary spectrum based delay estimation.
  BinaryDelayEstimatorFarend* binary_farend;
} DelayEstimatorFarend;

typedef struct {
  // Pointers to mean values of spectrum.
  SpectrumType* mean_near_spectrum;
  // `mean_near_spectrum` initialization indicator.
  int near_spectrum_initialized;

  int spectrum_size;

  // Binary spectrum based delay estimator
  BinaryDelayEstimator* binary_handle;
} DelayEstimator;

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_UTILITY_DELAY_ESTIMATOR_INTERNAL_H_
