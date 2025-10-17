/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#include "modules/audio_processing/aec3/frame_blocker.h"

#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/checks.h"

namespace webrtc {

FrameBlocker::FrameBlocker(size_t num_bands, size_t num_channels)
    : num_bands_(num_bands),
      num_channels_(num_channels),
      buffer_(num_bands_, std::vector<std::vector<float>>(num_channels)) {
  RTC_DCHECK_LT(0, num_bands);
  RTC_DCHECK_LT(0, num_channels);
  for (auto& band : buffer_) {
    for (auto& channel : band) {
      channel.reserve(kBlockSize);
      RTC_DCHECK(channel.empty());
    }
  }
}

FrameBlocker::~FrameBlocker() = default;

void FrameBlocker::InsertSubFrameAndExtractBlock(
    const std::vector<std::vector<rtc::ArrayView<float>>>& sub_frame,
    Block* block) {
  RTC_DCHECK(block);
  RTC_DCHECK_EQ(num_bands_, block->NumBands());
  RTC_DCHECK_EQ(num_bands_, sub_frame.size());
  for (size_t band = 0; band < num_bands_; ++band) {
    RTC_DCHECK_EQ(num_channels_, block->NumChannels());
    RTC_DCHECK_EQ(num_channels_, sub_frame[band].size());
    for (size_t channel = 0; channel < num_channels_; ++channel) {
      RTC_DCHECK_GE(kBlockSize - 16, buffer_[band][channel].size());
      RTC_DCHECK_EQ(kSubFrameLength, sub_frame[band][channel].size());
      const int samples_to_block = kBlockSize - buffer_[band][channel].size();
      std::copy(buffer_[band][channel].begin(), buffer_[band][channel].end(),
                block->begin(band, channel));
      std::copy(sub_frame[band][channel].begin(),
                sub_frame[band][channel].begin() + samples_to_block,
                block->begin(band, channel) + kBlockSize - samples_to_block);
      buffer_[band][channel].clear();
      buffer_[band][channel].insert(
          buffer_[band][channel].begin(),
          sub_frame[band][channel].begin() + samples_to_block,
          sub_frame[band][channel].end());
    }
  }
}

bool FrameBlocker::IsBlockAvailable() const {
  return kBlockSize == buffer_[0][0].size();
}

void FrameBlocker::ExtractBlock(Block* block) {
  RTC_DCHECK(block);
  RTC_DCHECK_EQ(num_bands_, block->NumBands());
  RTC_DCHECK_EQ(num_channels_, block->NumChannels());
  RTC_DCHECK(IsBlockAvailable());
  for (size_t band = 0; band < num_bands_; ++band) {
    for (size_t channel = 0; channel < num_channels_; ++channel) {
      RTC_DCHECK_EQ(kBlockSize, buffer_[band][channel].size());
      std::copy(buffer_[band][channel].begin(), buffer_[band][channel].end(),
                block->begin(band, channel));
      buffer_[band][channel].clear();
    }
  }
}

}  // namespace webrtc
