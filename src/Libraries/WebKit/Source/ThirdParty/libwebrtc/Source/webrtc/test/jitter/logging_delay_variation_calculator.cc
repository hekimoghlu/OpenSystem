/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include "test/jitter/logging_delay_variation_calculator.h"

#include "api/test/metrics/global_metrics_logger_and_exporter.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace test {

void LoggingDelayVariationCalculator::Insert(
    uint32_t rtp_timestamp,
    Timestamp arrival_time,
    DataSize size,
    std::optional<int> spatial_layer,
    std::optional<int> temporal_layer,
    std::optional<VideoFrameType> frame_type) {
  calc_.Insert(rtp_timestamp, arrival_time, size, spatial_layer, temporal_layer,
               frame_type);
}

void LoggingDelayVariationCalculator::LogMetrics() const {
  const DelayVariationCalculator::TimeSeries& time_series = calc_.time_series();

  if (!time_series.rtp_timestamps.IsEmpty()) {
    logger_->LogMetric("rtp_timestamp", log_type_, time_series.rtp_timestamps,
                       Unit::kUnitless, ImprovementDirection::kNeitherIsBetter);
  }
  if (!time_series.arrival_times_ms.IsEmpty()) {
    logger_->LogMetric("arrival_time", log_type_, time_series.arrival_times_ms,
                       Unit::kMilliseconds,
                       ImprovementDirection::kNeitherIsBetter);
  }
  if (!time_series.sizes_bytes.IsEmpty()) {
    logger_->LogMetric("size_bytes", log_type_, time_series.sizes_bytes,
                       Unit::kBytes, ImprovementDirection::kNeitherIsBetter);
  }
  if (!time_series.inter_departure_times_ms.IsEmpty()) {
    logger_->LogMetric(
        "inter_departure_time", log_type_, time_series.inter_departure_times_ms,
        Unit::kMilliseconds, ImprovementDirection::kNeitherIsBetter);
  }
  if (!time_series.inter_arrival_times_ms.IsEmpty()) {
    logger_->LogMetric("inter_arrival_time", log_type_,
                       time_series.inter_arrival_times_ms, Unit::kMilliseconds,
                       ImprovementDirection::kNeitherIsBetter);
  }
  if (!time_series.inter_delay_variations_ms.IsEmpty()) {
    logger_->LogMetric("inter_delay_variation", log_type_,
                       time_series.inter_delay_variations_ms,
                       Unit::kMilliseconds,
                       ImprovementDirection::kNeitherIsBetter);
  }
  if (!time_series.inter_size_variations_bytes.IsEmpty()) {
    logger_->LogMetric("inter_size_variation", log_type_,
                       time_series.inter_size_variations_bytes, Unit::kBytes,
                       ImprovementDirection::kNeitherIsBetter);
  }
}

}  // namespace test
}  // namespace webrtc
