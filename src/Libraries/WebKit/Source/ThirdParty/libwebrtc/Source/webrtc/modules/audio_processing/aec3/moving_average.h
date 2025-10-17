/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MOVING_AVERAGE_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MOVING_AVERAGE_H_

#include <stddef.h>

#include <vector>

#include "api/array_view.h"

namespace webrtc {
namespace aec3 {

class MovingAverage {
 public:
  // Creates an instance of MovingAverage that accepts inputs of length num_elem
  // and averages over mem_len inputs.
  MovingAverage(size_t num_elem, size_t mem_len);
  ~MovingAverage();

  // Computes the average of input and mem_len-1 previous inputs and stores the
  // result in output.
  void Average(rtc::ArrayView<const float> input, rtc::ArrayView<float> output);

 private:
  const size_t num_elem_;
  const size_t mem_len_;
  const float scaling_;
  std::vector<float> memory_;
  size_t mem_index_;
};

}  // namespace aec3
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MOVING_AVERAGE_H_
