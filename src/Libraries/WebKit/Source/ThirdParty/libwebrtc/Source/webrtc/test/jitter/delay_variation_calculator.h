/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#ifndef TEST_JITTER_DELAY_VARIATION_CALCULATOR_H_
#define TEST_JITTER_DELAY_VARIATION_CALCULATOR_H_

#include <stdint.h>

#include <map>
#include <optional>
#include <string>

#include "api/numerics/samples_stats_counter.h"
#include "api/test/metrics/metrics_logger.h"
#include "api/units/data_size.h"
#include "api/units/timestamp.h"
#include "api/video/video_frame_type.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {
namespace test {

// Helper class for calculating different delay variation statistics for "RTP
// frame arrival events". One use case is gathering statistics from
// RtcEventLogs. Another use case is online logging of data in test calls.
class DelayVariationCalculator {
 public:
  struct TimeSeries {
    // Time series of RTP timestamps `t(n)` for each frame `n`.
    SamplesStatsCounter rtp_timestamps;
    // Time series of local arrival timestamps `r(n)` for each frame.
    SamplesStatsCounter arrival_times_ms;
    // Time series of sizes `s(n)` for each frame.
    SamplesStatsCounter sizes_bytes;
    // Time series of `d_t(n) = t(n) - t(n-1)` for each frame.
    SamplesStatsCounter inter_departure_times_ms;
    // Time series of `d_r(n) = r(n) - r(n-1)` for each frame.
    SamplesStatsCounter inter_arrival_times_ms;
    // Time series of `d_r(n) - d_t(n) = (r(n) - r(n-1)) - (t(n) - t(n-1))`
    // for each frame.
    SamplesStatsCounter inter_delay_variations_ms;
    // Time series of `s(n) - s(n-1)`, for each frame.
    SamplesStatsCounter inter_size_variations_bytes;
  };

  DelayVariationCalculator() = default;
  ~DelayVariationCalculator() = default;

  void Insert(uint32_t rtp_timestamp,
              Timestamp arrival_time,
              DataSize size,
              std::optional<int> spatial_layer = std::nullopt,
              std::optional<int> temporal_layer = std::nullopt,
              std::optional<VideoFrameType> frame_type = std::nullopt);

  const TimeSeries& time_series() const { return time_series_; }

 private:
  struct Frame {
    uint32_t rtp_timestamp;
    int64_t unwrapped_rtp_timestamp;
    Timestamp arrival_time;
    DataSize size;
    std::optional<int> spatial_layer;
    std::optional<int> temporal_layer;
    std::optional<VideoFrameType> frame_type;
  };
  using MetadataT = std::map<std::string, std::string>;

  void InsertFirstFrame(const Frame& frame,
                        Timestamp sample_time,
                        MetadataT sample_metadata);
  void InsertFrame(const Frame& frame,
                   Timestamp sample_time,
                   MetadataT sample_metadata);

  MetadataT BuildMetadata(const Frame& frame);

  RtpTimestampUnwrapper unwrapper_;
  std::optional<Frame> prev_frame_ = std::nullopt;
  TimeSeries time_series_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_JITTER_DELAY_VARIATION_CALCULATOR_H_
