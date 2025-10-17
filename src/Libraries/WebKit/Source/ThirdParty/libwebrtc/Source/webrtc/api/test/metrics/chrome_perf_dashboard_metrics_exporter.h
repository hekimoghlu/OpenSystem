/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#ifndef API_TEST_METRICS_CHROME_PERF_DASHBOARD_METRICS_EXPORTER_H_
#define API_TEST_METRICS_CHROME_PERF_DASHBOARD_METRICS_EXPORTER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/test/metrics/metric.h"
#include "api/test/metrics/metrics_exporter.h"

namespace webrtc {
namespace test {

// Exports all collected metrics in the Chrome Perf Dashboard proto format.
class ChromePerfDashboardMetricsExporter : public MetricsExporter {
 public:
  // `export_file_path` - path where the proto file will be written.
  explicit ChromePerfDashboardMetricsExporter(
      absl::string_view export_file_path);
  ~ChromePerfDashboardMetricsExporter() override = default;

  bool Export(rtc::ArrayView<const Metric> metrics) override;

 private:
  const std::string export_file_path_;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_METRICS_CHROME_PERF_DASHBOARD_METRICS_EXPORTER_H_
