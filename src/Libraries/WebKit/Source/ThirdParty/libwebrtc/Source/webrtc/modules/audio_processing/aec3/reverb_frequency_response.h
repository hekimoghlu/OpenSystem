/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_REVERB_FREQUENCY_RESPONSE_H_
#define MODULES_AUDIO_PROCESSING_AEC3_REVERB_FREQUENCY_RESPONSE_H_

#include <array>
#include <optional>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"

namespace webrtc {

// Class for updating the frequency response for the reverb.
class ReverbFrequencyResponse {
 public:
  explicit ReverbFrequencyResponse(
      bool use_conservative_tail_frequency_response);
  ~ReverbFrequencyResponse();

  // Updates the frequency response estimate of the reverb.
  void Update(const std::vector<std::array<float, kFftLengthBy2Plus1>>&
                  frequency_response,
              int filter_delay_blocks,
              const std::optional<float>& linear_filter_quality,
              bool stationary_block);

  // Returns the estimated frequency response for the reverb.
  rtc::ArrayView<const float> FrequencyResponse() const {
    return tail_response_;
  }

 private:
  void Update(const std::vector<std::array<float, kFftLengthBy2Plus1>>&
                  frequency_response,
              int filter_delay_blocks,
              float linear_filter_quality);

  const bool use_conservative_tail_frequency_response_;
  float average_decay_ = 0.f;
  std::array<float, kFftLengthBy2Plus1> tail_response_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_REVERB_FREQUENCY_RESPONSE_H_
