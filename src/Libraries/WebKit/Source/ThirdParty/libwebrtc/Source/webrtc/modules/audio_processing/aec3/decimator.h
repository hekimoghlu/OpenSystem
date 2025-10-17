/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_DECIMATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_DECIMATOR_H_

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/utility/cascaded_biquad_filter.h"

namespace webrtc {

// Provides functionality for decimating a signal.
class Decimator {
 public:
  explicit Decimator(size_t down_sampling_factor);

  Decimator(const Decimator&) = delete;
  Decimator& operator=(const Decimator&) = delete;

  // Downsamples the signal.
  void Decimate(rtc::ArrayView<const float> in, rtc::ArrayView<float> out);

 private:
  const size_t down_sampling_factor_;
  CascadedBiQuadFilter anti_aliasing_filter_;
  CascadedBiQuadFilter noise_reduction_filter_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_DECIMATOR_H_
