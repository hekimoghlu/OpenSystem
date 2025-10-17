/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#ifndef API_VIDEO_VIDEO_BITRATE_ALLOCATION_H_
#define API_VIDEO_VIDEO_BITRATE_ALLOCATION_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "api/video/video_codec_constants.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Class that describes how video bitrate, in bps, is allocated across temporal
// and spatial layers. Not that bitrates are NOT cumulative. Depending on if
// layers are dependent or not, it is up to the user to aggregate.
// For each index, the bitrate can also both set and unset. This is used with a
// set bps = 0 to signal an explicit "turn off" signal.
class RTC_EXPORT VideoBitrateAllocation {
 public:
  static constexpr uint32_t kMaxBitrateBps =
      std::numeric_limits<uint32_t>::max();
  VideoBitrateAllocation();

  bool SetBitrate(size_t spatial_index,
                  size_t temporal_index,
                  uint32_t bitrate_bps);

  bool HasBitrate(size_t spatial_index, size_t temporal_index) const;

  uint32_t GetBitrate(size_t spatial_index, size_t temporal_index) const;

  // Whether the specific spatial layers has the bitrate set in any of its
  // temporal layers.
  bool IsSpatialLayerUsed(size_t spatial_index) const;

  // Get the sum of all the temporal layer for a specific spatial layer.
  uint32_t GetSpatialLayerSum(size_t spatial_index) const;

  // Sum of bitrates of temporal layers, from layer 0 to `temporal_index`
  // inclusive, of specified spatial layer `spatial_index`. Bitrates of lower
  // spatial layers are not included.
  uint32_t GetTemporalLayerSum(size_t spatial_index,
                               size_t temporal_index) const;

  // Returns a vector of the temporal layer bitrates for the specific spatial
  // layer. Length of the returned vector is cropped to the highest temporal
  // layer with a defined bitrate.
  std::vector<uint32_t> GetTemporalLayerAllocation(size_t spatial_index) const;

  // Returns one VideoBitrateAllocation for each spatial layer. This is used to
  // configure simulcast streams. Note that the length of the returned vector is
  // always kMaxSpatialLayers, the optional is unset for unused layers.
  std::vector<std::optional<VideoBitrateAllocation>> GetSimulcastAllocations()
      const;

  uint32_t get_sum_bps() const { return sum_; }  // Sum of all bitrates.
  uint32_t get_sum_kbps() const {
    // Round down to not exceed the allocated bitrate.
    return sum_ / 1000;
  }

  bool operator==(const VideoBitrateAllocation& other) const;
  inline bool operator!=(const VideoBitrateAllocation& other) const {
    return !(*this == other);
  }

  std::string ToString() const;

  // Indicates if the allocation has some layers/streams disabled due to
  // low available bandwidth.
  void set_bw_limited(bool limited) { is_bw_limited_ = limited; }
  bool is_bw_limited() const { return is_bw_limited_; }

 private:
  uint32_t sum_;
  std::optional<uint32_t> bitrates_[kMaxSpatialLayers][kMaxTemporalStreams];
  bool is_bw_limited_;
};

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_BITRATE_ALLOCATION_H_
