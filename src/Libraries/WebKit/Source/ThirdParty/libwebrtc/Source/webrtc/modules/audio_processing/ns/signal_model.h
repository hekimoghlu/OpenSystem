/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_H_
#define MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_H_

#include <array>

#include "modules/audio_processing/ns/ns_common.h"

namespace webrtc {

struct SignalModel {
  SignalModel();
  SignalModel(const SignalModel&) = delete;
  SignalModel& operator=(const SignalModel&) = delete;

  float lrt;
  float spectral_diff;
  float spectral_flatness;
  // Log LRT factor with time-smoothing.
  std::array<float, kFftSizeBy2Plus1> avg_log_lrt;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_H_
