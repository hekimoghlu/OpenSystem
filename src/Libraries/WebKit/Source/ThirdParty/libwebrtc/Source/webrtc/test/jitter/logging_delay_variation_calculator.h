/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#ifndef TEST_JITTER_LOGGING_DELAY_VARIATION_CALCULATOR_H_
#define TEST_JITTER_LOGGING_DELAY_VARIATION_CALCULATOR_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/test/metrics/global_metrics_logger_and_exporter.h"
#include "api/units/data_size.h"
#include "test/jitter/delay_variation_calculator.h"

namespace webrtc {
namespace test {

// This class logs the results from a `DelayVariationCalculator`
// to a metrics logger. For ease of integration, logging happens at
// object destruction.
class LoggingDelayVariationCalculator {
 public:
  LoggingDelayVariationCalculator(
      absl::string_view log_type,
      DefaultMetricsLogger* logger = GetGlobalMetricsLogger())
      : log_type_(log_type), logger_(logger) {}
  ~LoggingDelayVariationCalculator() { LogMetrics(); }

  void Insert(uint32_t rtp_timestamp,
              Timestamp arrival_time,
              DataSize size,
              std::optional<int> spatial_layer = std::nullopt,
              std::optional<int> temporal_layer = std::nullopt,
              std::optional<VideoFrameType> frame_type = std::nullopt);

 private:
  void LogMetrics() const;

  const std::string log_type_;
  DefaultMetricsLogger* const logger_;
  DelayVariationCalculator calc_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_JITTER_LOGGING_DELAY_VARIATION_CALCULATOR_H_
