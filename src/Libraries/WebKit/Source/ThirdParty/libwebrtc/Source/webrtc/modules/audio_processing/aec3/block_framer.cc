/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "modules/audio_processing/aec3/block_framer.h"

#include <algorithm>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "rtc_base/checks.h"

namespace webrtc {

BlockFramer::BlockFramer(size_t num_bands, size_t num_channels)
    : num_bands_(num_bands),
      num_channels_(num_channels),
      buffer_(num_bands_,
              std::vector<std::vector<float>>(
                  num_channels,
                  std::vector<float>(kBlockSize, 0.f))) {
  RTC_DCHECK_LT(0, num_bands);
  RTC_DCHECK_LT(0, num_channels);
}

BlockFramer::~BlockFramer() = default;

// All the constants are chosen so that the buffer is either empty or has enough
// samples for InsertBlockAndExtractSubFrame to produce a frame. In order to
// achieve this, the InsertBlockAndExtractSubFrame and InsertBlock methods need
// to be called in the correct order.
void BlockFramer::InsertBlock(const Block& block) {
  RTC_DCHECK_EQ(num_bands_, block.NumBands());
  RTC_DCHECK_EQ(num_channels_, block.NumChannels());
  for (size_t band = 0; band < num_bands_; ++band) {
    for (size_t channel = 0; channel < num_channels_; ++channel) {
      RTC_DCHECK_EQ(0, buffer_[band][channel].size());

      buffer_[band][channel].insert(buffer_[band][channel].begin(),
                                    block.begin(band, channel),
                                    block.end(band, channel));
    }
  }
}

void BlockFramer::InsertBlockAndExtractSubFrame(
    const Block& block,
    std::vector<std::vector<rtc::ArrayView<float>>>* sub_frame) {
  RTC_DCHECK(sub_frame);
  RTC_DCHECK_EQ(num_bands_, block.NumBands());
  RTC_DCHECK_EQ(num_channels_, block.NumChannels());
  RTC_DCHECK_EQ(num_bands_, sub_frame->size());
  for (size_t band = 0; band < num_bands_; ++band) {
    RTC_DCHECK_EQ(num_channels_, (*sub_frame)[0].size());
    for (size_t channel = 0; channel < num_channels_; ++channel) {
      RTC_DCHECK_LE(kSubFrameLength,
                    buffer_[band][channel].size() + kBlockSize);
      RTC_DCHECK_GE(kBlockSize, buffer_[band][channel].size());
      RTC_DCHECK_EQ(kSubFrameLength, (*sub_frame)[band][channel].size());

      const int samples_to_frame =
          kSubFrameLength - buffer_[band][channel].size();
      std::copy(buffer_[band][channel].begin(), buffer_[band][channel].end(),
                (*sub_frame)[band][channel].begin());
      std::copy(
          block.begin(band, channel),
          block.begin(band, channel) + samples_to_frame,
          (*sub_frame)[band][channel].begin() + buffer_[band][channel].size());
      buffer_[band][channel].clear();
      buffer_[band][channel].insert(
          buffer_[band][channel].begin(),
          block.begin(band, channel) + samples_to_frame,
          block.end(band, channel));
    }
  }
}

}  // namespace webrtc
