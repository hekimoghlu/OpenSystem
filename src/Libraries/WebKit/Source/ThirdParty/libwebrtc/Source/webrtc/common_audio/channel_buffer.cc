/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#include "common_audio/channel_buffer.h"

#include <cstdint>

#include "common_audio/include/audio_util.h"
#include "rtc_base/checks.h"

namespace webrtc {

IFChannelBuffer::IFChannelBuffer(size_t num_frames,
                                 size_t num_channels,
                                 size_t num_bands)
    : ivalid_(true),
      ibuf_(num_frames, num_channels, num_bands),
      fvalid_(true),
      fbuf_(num_frames, num_channels, num_bands) {}

IFChannelBuffer::~IFChannelBuffer() = default;

ChannelBuffer<int16_t>* IFChannelBuffer::ibuf() {
  RefreshI();
  fvalid_ = false;
  return &ibuf_;
}

ChannelBuffer<float>* IFChannelBuffer::fbuf() {
  RefreshF();
  ivalid_ = false;
  return &fbuf_;
}

const ChannelBuffer<int16_t>* IFChannelBuffer::ibuf_const() const {
  RefreshI();
  return &ibuf_;
}

const ChannelBuffer<float>* IFChannelBuffer::fbuf_const() const {
  RefreshF();
  return &fbuf_;
}

void IFChannelBuffer::RefreshF() const {
  if (!fvalid_) {
    RTC_DCHECK(ivalid_);
    fbuf_.set_num_channels(ibuf_.num_channels());
    const int16_t* const* int_channels = ibuf_.channels();
    float* const* float_channels = fbuf_.channels();
    for (size_t i = 0; i < ibuf_.num_channels(); ++i) {
      for (size_t j = 0; j < ibuf_.num_frames(); ++j) {
        float_channels[i][j] = int_channels[i][j];
      }
    }
    fvalid_ = true;
  }
}

void IFChannelBuffer::RefreshI() const {
  if (!ivalid_) {
    RTC_DCHECK(fvalid_);
    int16_t* const* int_channels = ibuf_.channels();
    ibuf_.set_num_channels(fbuf_.num_channels());
    const float* const* float_channels = fbuf_.channels();
    for (size_t i = 0; i < fbuf_.num_channels(); ++i) {
      FloatS16ToS16(float_channels[i], ibuf_.num_frames(), int_channels[i]);
    }
    ivalid_ = true;
  }
}

}  // namespace webrtc
