/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "modules/audio_processing/agc2/clipping_predictor_level_buffer.h"

#include <algorithm>
#include <cmath>

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

bool ClippingPredictorLevelBuffer::Level::operator==(const Level& level) const {
  constexpr float kEpsilon = 1e-6f;
  return std::fabs(average - level.average) < kEpsilon &&
         std::fabs(max - level.max) < kEpsilon;
}

ClippingPredictorLevelBuffer::ClippingPredictorLevelBuffer(int capacity)
    : tail_(-1), size_(0), data_(std::max(1, capacity)) {
  if (capacity > kMaxCapacity) {
    RTC_LOG(LS_WARNING) << "[agc]: ClippingPredictorLevelBuffer exceeds the "
                        << "maximum allowed capacity. Capacity: " << capacity;
  }
  RTC_DCHECK(!data_.empty());
}

void ClippingPredictorLevelBuffer::Reset() {
  tail_ = -1;
  size_ = 0;
}

void ClippingPredictorLevelBuffer::Push(Level level) {
  ++tail_;
  if (tail_ == Capacity()) {
    tail_ = 0;
  }
  if (size_ < Capacity()) {
    size_++;
  }
  data_[tail_] = level;
}

// TODO(bugs.webrtc.org/12774): Optimize partial computation for long buffers.
std::optional<ClippingPredictorLevelBuffer::Level>
ClippingPredictorLevelBuffer::ComputePartialMetrics(int delay,
                                                    int num_items) const {
  RTC_DCHECK_GE(delay, 0);
  RTC_DCHECK_LT(delay, Capacity());
  RTC_DCHECK_GT(num_items, 0);
  RTC_DCHECK_LE(num_items, Capacity());
  RTC_DCHECK_LE(delay + num_items, Capacity());
  if (delay + num_items > Size()) {
    return std::nullopt;
  }
  float sum = 0.0f;
  float max = 0.0f;
  for (int i = 0; i < num_items && i < Size(); ++i) {
    int idx = tail_ - delay - i;
    if (idx < 0) {
      idx += Capacity();
    }
    sum += data_[idx].average;
    max = std::fmax(data_[idx].max, max);
  }
  return std::optional<Level>({sum / static_cast<float>(num_items), max});
}

}  // namespace webrtc
