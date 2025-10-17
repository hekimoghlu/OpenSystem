/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#include "modules/audio_processing/aec3/moving_average.h"

#include <algorithm>
#include <functional>

#include "rtc_base/checks.h"

namespace webrtc {
namespace aec3 {

MovingAverage::MovingAverage(size_t num_elem, size_t mem_len)
    : num_elem_(num_elem),
      mem_len_(mem_len - 1),
      scaling_(1.0f / static_cast<float>(mem_len)),
      memory_(num_elem * mem_len_, 0.f),
      mem_index_(0) {
  RTC_DCHECK(num_elem_ > 0);
  RTC_DCHECK(mem_len > 0);
}

MovingAverage::~MovingAverage() = default;

void MovingAverage::Average(rtc::ArrayView<const float> input,
                            rtc::ArrayView<float> output) {
  RTC_DCHECK(input.size() == num_elem_);
  RTC_DCHECK(output.size() == num_elem_);

  // Sum all contributions.
  std::copy(input.begin(), input.end(), output.begin());
  for (auto i = memory_.begin(); i < memory_.end(); i += num_elem_) {
    std::transform(i, i + num_elem_, output.begin(), output.begin(),
                   std::plus<float>());
  }

  // Divide by mem_len_.
  for (float& o : output) {
    o *= scaling_;
  }

  // Update memory.
  if (mem_len_ > 0) {
    std::copy(input.begin(), input.end(),
              memory_.begin() + mem_index_ * num_elem_);
    mem_index_ = (mem_index_ + 1) % mem_len_;
  }
}

}  // namespace aec3
}  // namespace webrtc
