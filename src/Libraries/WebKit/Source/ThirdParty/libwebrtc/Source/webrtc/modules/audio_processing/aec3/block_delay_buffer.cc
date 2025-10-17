/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include "modules/audio_processing/aec3/block_delay_buffer.h"

#include "api/array_view.h"
#include "rtc_base/checks.h"

namespace webrtc {

BlockDelayBuffer::BlockDelayBuffer(size_t num_channels,
                                   size_t num_bands,
                                   size_t frame_length,
                                   size_t delay_samples)
    : frame_length_(frame_length),
      delay_(delay_samples),
      buf_(num_channels,
           std::vector<std::vector<float>>(num_bands,
                                           std::vector<float>(delay_, 0.f))) {}

BlockDelayBuffer::~BlockDelayBuffer() = default;

void BlockDelayBuffer::DelaySignal(AudioBuffer* frame) {
  RTC_DCHECK_EQ(buf_.size(), frame->num_channels());
  if (delay_ == 0) {
    return;
  }

  const size_t num_bands = buf_[0].size();
  const size_t num_channels = buf_.size();

  const size_t i_start = last_insert_;
  size_t i = 0;
  for (size_t ch = 0; ch < num_channels; ++ch) {
    RTC_DCHECK_EQ(buf_[ch].size(), frame->num_bands());
    RTC_DCHECK_EQ(buf_[ch].size(), num_bands);
    rtc::ArrayView<float* const> frame_ch(frame->split_bands(ch), num_bands);
    const size_t delay = delay_;

    for (size_t band = 0; band < num_bands; ++band) {
      RTC_DCHECK_EQ(delay_, buf_[ch][band].size());
      i = i_start;

      // Offloading these pointers and class variables to local variables allows
      // the compiler to optimize the below loop when compiling with
      // '-fno-strict-aliasing'.
      float* buf_ch_band = buf_[ch][band].data();
      float* frame_ch_band = frame_ch[band];

      for (size_t k = 0, frame_length = frame_length_; k < frame_length; ++k) {
        const float tmp = buf_ch_band[i];
        buf_ch_band[i] = frame_ch_band[k];
        frame_ch_band[k] = tmp;

        i = i < delay - 1 ? i + 1 : 0;
      }
    }
  }

  last_insert_ = i;
}

}  // namespace webrtc
