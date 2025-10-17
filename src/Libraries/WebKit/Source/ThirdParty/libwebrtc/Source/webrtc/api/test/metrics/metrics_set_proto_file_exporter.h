/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#ifndef API_TEST_METRICS_METRICS_SET_PROTO_FILE_EXPORTER_H_
#define API_TEST_METRICS_METRICS_SET_PROTO_FILE_EXPORTER_H_

#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/test/metrics/metric.h"
#include "api/test/metrics/metrics_exporter.h"

namespace webrtc {
namespace test {

// Exports all collected metrics to the proto file using
// `webrtc::test_metrics::MetricsSet` format.
class MetricsSetProtoFileExporter : public MetricsExporter {
 public:
  struct Options {
    explicit Options(absl::string_view export_file_path);
    Options(absl::string_view export_file_path, bool export_whole_time_series);
    Options(absl::string_view export_file_path,
            std::map<std::string, std::string> metadata);

    // File to export proto.
    std::string export_file_path;
    // If true will write all time series values to the output proto file,
    // otherwise will write stats only.
    bool export_whole_time_series = true;
    // Metadata associated to the whole MetricsSet.
    std::map<std::string, std::string> metadata;
  };

  explicit MetricsSetProtoFileExporter(const Options& options)
      : options_(options) {}

  MetricsSetProtoFileExporter(const MetricsSetProtoFileExporter&) = delete;
  MetricsSetProtoFileExporter& operator=(const MetricsSetProtoFileExporter&) =
      delete;

  bool Export(rtc::ArrayView<const Metric> metrics) override;

 private:
  const Options options_;
};

}  // namespace test
}  // namespace webrtc

#endif  // API_TEST_METRICS_METRICS_SET_PROTO_FILE_EXPORTER_H_
