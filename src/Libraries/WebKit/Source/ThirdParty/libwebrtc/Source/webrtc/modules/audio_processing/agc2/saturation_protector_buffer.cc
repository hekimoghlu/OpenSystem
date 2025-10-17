/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#include "modules/audio_processing/agc2/saturation_protector_buffer.h"

#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_compare.h"

namespace webrtc {

SaturationProtectorBuffer::SaturationProtectorBuffer() = default;

SaturationProtectorBuffer::~SaturationProtectorBuffer() = default;

bool SaturationProtectorBuffer::operator==(
    const SaturationProtectorBuffer& b) const {
  RTC_DCHECK_LE(size_, buffer_.size());
  RTC_DCHECK_LE(b.size_, b.buffer_.size());
  if (size_ != b.size_) {
    return false;
  }
  for (int i = 0, i0 = FrontIndex(), i1 = b.FrontIndex(); i < size_;
       ++i, ++i0, ++i1) {
    if (buffer_[i0 % buffer_.size()] != b.buffer_[i1 % b.buffer_.size()]) {
      return false;
    }
  }
  return true;
}

int SaturationProtectorBuffer::Capacity() const {
  return buffer_.size();
}

int SaturationProtectorBuffer::Size() const {
  return size_;
}

void SaturationProtectorBuffer::Reset() {
  next_ = 0;
  size_ = 0;
}

void SaturationProtectorBuffer::PushBack(float v) {
  RTC_DCHECK_GE(next_, 0);
  RTC_DCHECK_GE(size_, 0);
  RTC_DCHECK_LT(next_, buffer_.size());
  RTC_DCHECK_LE(size_, buffer_.size());
  buffer_[next_++] = v;
  if (rtc::SafeEq(next_, buffer_.size())) {
    next_ = 0;
  }
  if (rtc::SafeLt(size_, buffer_.size())) {
    size_++;
  }
}

std::optional<float> SaturationProtectorBuffer::Front() const {
  if (size_ == 0) {
    return std::nullopt;
  }
  RTC_DCHECK_LT(FrontIndex(), buffer_.size());
  return buffer_[FrontIndex()];
}

int SaturationProtectorBuffer::FrontIndex() const {
  return rtc::SafeEq(size_, buffer_.size()) ? next_ : 0;
}

}  // namespace webrtc
