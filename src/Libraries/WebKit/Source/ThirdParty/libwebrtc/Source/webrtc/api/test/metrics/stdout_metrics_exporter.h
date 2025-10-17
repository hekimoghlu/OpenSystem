/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#ifndef API_TEST_METRICS_STDOUT_METRICS_EXPORTER_H_
#define API_TEST_METRICS_STDOUT_METRICS_EXPORTER_H_

#include "api/array_view.h"
#include "api/test/metrics/metric.h"
#include "api/test/metrics/metrics_exporter.h"

namespace webrtc {
namespace test {

// Exports all collected metrics to stdout.
class StdoutMetricsExporter : public MetricsExporter {
 public:
  StdoutMetricsExporter();
  ~StdoutMetricsExporter() override = default;

  StdoutMetricsExporter(const StdoutMetricsExporter&) = delete;
  StdoutMetricsExporter& operator=(const StdoutMetricsExporter&) = delete;

  bool Export(rtc::ArrayView<const Metric> metrics) override;

 private:
  void PrintMetric(const Metric& metric);

  FILE* const output_;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_METRICS_STDOUT_METRICS_EXPORTER_H_
