/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#include "modules/audio_processing/echo_detector/circular_buffer.h"

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {

CircularBuffer::CircularBuffer(size_t size) : buffer_(size) {}
CircularBuffer::~CircularBuffer() = default;

void CircularBuffer::Push(float value) {
  buffer_[next_insertion_index_] = value;
  ++next_insertion_index_;
  next_insertion_index_ %= buffer_.size();
  RTC_DCHECK_LT(next_insertion_index_, buffer_.size());
  nr_elements_in_buffer_ = std::min(nr_elements_in_buffer_ + 1, buffer_.size());
  RTC_DCHECK_LE(nr_elements_in_buffer_, buffer_.size());
}

std::optional<float> CircularBuffer::Pop() {
  if (nr_elements_in_buffer_ == 0) {
    return std::nullopt;
  }
  const size_t index =
      (buffer_.size() + next_insertion_index_ - nr_elements_in_buffer_) %
      buffer_.size();
  RTC_DCHECK_LT(index, buffer_.size());
  --nr_elements_in_buffer_;
  return buffer_[index];
}

void CircularBuffer::Clear() {
  std::fill(buffer_.begin(), buffer_.end(), 0.f);
  next_insertion_index_ = 0;
  nr_elements_in_buffer_ = 0;
}

}  // namespace webrtc
