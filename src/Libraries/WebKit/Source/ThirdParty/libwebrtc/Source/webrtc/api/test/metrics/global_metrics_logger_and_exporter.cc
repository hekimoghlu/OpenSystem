/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#include "api/test/metrics/global_metrics_logger_and_exporter.h"

#include <memory>
#include <utility>
#include <vector>

#include "api/test/metrics/metrics_exporter.h"
#include "api/test/metrics/metrics_logger.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
namespace test {

DefaultMetricsLogger* GetGlobalMetricsLogger() {
  static DefaultMetricsLogger* logger_ =
      new DefaultMetricsLogger(Clock::GetRealTimeClock());
  return logger_;
}

bool ExportPerfMetric(MetricsLogger& logger,
                      std::vector<std::unique_ptr<MetricsExporter>> exporters) {
  std::vector<Metric> metrics = logger.GetCollectedMetrics();
  bool success = true;
  for (auto& exporter : exporters) {
    bool export_result = exporter->Export(metrics);
    success = success && export_result;
  }
  return success;
}

}  // namespace test
}  // namespace webrtc
