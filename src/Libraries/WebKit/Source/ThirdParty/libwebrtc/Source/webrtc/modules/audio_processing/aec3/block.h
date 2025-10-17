/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_BLOCK_H_
#define MODULES_AUDIO_PROCESSING_AEC3_BLOCK_H_

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"

namespace webrtc {

// Contains one or more channels of 4 milliseconds of audio data.
// The audio is split in one or more frequency bands, each with a sampling
// rate of 16 kHz.
class Block {
 public:
  Block(int num_bands, int num_channels, float default_value = 0.0f)
      : num_bands_(num_bands),
        num_channels_(num_channels),
        data_(num_bands * num_channels * kBlockSize, default_value) {}

  // Returns the number of bands.
  int NumBands() const { return num_bands_; }

  // Returns the number of channels.
  int NumChannels() const { return num_channels_; }

  // Modifies the number of channels and sets all samples to zero.
  void SetNumChannels(int num_channels) {
    num_channels_ = num_channels;
    data_.resize(num_bands_ * num_channels_ * kBlockSize);
    std::fill(data_.begin(), data_.end(), 0.0f);
  }

  // Iterators for accessing the data.
  auto begin(int band, int channel) {
    return data_.begin() + GetIndex(band, channel);
  }

  auto begin(int band, int channel) const {
    return data_.begin() + GetIndex(band, channel);
  }

  auto end(int band, int channel) { return begin(band, channel) + kBlockSize; }

  auto end(int band, int channel) const {
    return begin(band, channel) + kBlockSize;
  }

  // Access data via ArrayView.
  rtc::ArrayView<float, kBlockSize> View(int band, int channel) {
    return rtc::ArrayView<float, kBlockSize>(&data_[GetIndex(band, channel)],
                                             kBlockSize);
  }

  rtc::ArrayView<const float, kBlockSize> View(int band, int channel) const {
    return rtc::ArrayView<const float, kBlockSize>(
        &data_[GetIndex(band, channel)], kBlockSize);
  }

  // Lets two Blocks swap audio data.
  void Swap(Block& b) {
    std::swap(num_bands_, b.num_bands_);
    std::swap(num_channels_, b.num_channels_);
    data_.swap(b.data_);
  }

 private:
  // Returns the index of the first sample of the requested |band| and
  // |channel|.
  int GetIndex(int band, int channel) const {
    return (band * num_channels_ + channel) * kBlockSize;
  }

  int num_bands_;
  int num_channels_;
  std::vector<float> data_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_PROCESSING_AEC3_BLOCK_H_
