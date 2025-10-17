/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include "modules/audio_coding/neteq/buffer_level_filter.h"

#include <stdint.h>

#include <algorithm>

#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

BufferLevelFilter::BufferLevelFilter() {
  Reset();
}

void BufferLevelFilter::Reset() {
  filtered_current_level_ = 0;
  level_factor_ = 253;
}

void BufferLevelFilter::Update(size_t buffer_size_samples,
                               int time_stretched_samples) {
  // Filter:
  // `filtered_current_level_` = `level_factor_` * `filtered_current_level_` +
  //                            (1 - `level_factor_`) * `buffer_size_samples`
  // `level_factor_` and `filtered_current_level_` are in Q8.
  // `buffer_size_samples` is in Q0.
  const int64_t filtered_current_level =
      (level_factor_* int64_t{filtered_current_level_} >> 8) +
      (256 - level_factor_) * rtc::dchecked_cast<int64_t>(buffer_size_samples);

  // Account for time-scale operations (accelerate and pre-emptive expand) and
  // make sure that the filtered value remains non-negative.
  filtered_current_level_ = rtc::saturated_cast<int>(std::max<int64_t>(
      0, filtered_current_level - int64_t{time_stretched_samples} * (1 << 8)));
}

void BufferLevelFilter::SetFilteredBufferLevel(int buffer_size_samples) {
  filtered_current_level_ =
      rtc::saturated_cast<int>(int64_t{buffer_size_samples} * 256);
}

void BufferLevelFilter::SetTargetBufferLevel(int target_buffer_level_ms) {
  if (target_buffer_level_ms <= 20) {
    level_factor_ = 251;
  } else if (target_buffer_level_ms <= 60) {
    level_factor_ = 252;
  } else if (target_buffer_level_ms <= 140) {
    level_factor_ = 253;
  } else {
    level_factor_ = 254;
  }
}

}  // namespace webrtc
