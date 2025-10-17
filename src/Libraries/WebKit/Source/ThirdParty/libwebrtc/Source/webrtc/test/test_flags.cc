/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#include "test/test_flags.h"

#include <string>
#include <vector>

#include "absl/flags/flag.h"

ABSL_FLAG(std::string,
          force_fieldtrials,
          "",
          "Field trials control experimental feature code which can be forced. "
          "E.g. running with --force_fieldtrials=WebRTC-FooFeature/Enable/"
          " will assign the group Enable to field trial WebRTC-FooFeature.");

ABSL_FLAG(std::vector<std::string>,
          plot,
          {},
          "List of metrics that should be exported for plotting (if they are "
          "available). Example: psnr,ssim,encode_time. To plot all available "
          " metrics pass 'all' as flag value");

ABSL_FLAG(
    std::string,
    isolated_script_test_perf_output,
    "",
    "Path where the perf results should be stored in proto format described "
    "described by histogram.proto in "
    "https://chromium.googlesource.com/catapult/.");

ABSL_FLAG(std::string,
          webrtc_test_metrics_output_path,
          "",
          "Path where the test perf metrics should be stored using "
          "api/test/metrics/metric.proto proto format. File will contain "
          "MetricsSet as a root proto. On iOS, this MUST be a file name "
          "and the file will be stored under NSDocumentDirectory.");

ABSL_FLAG(bool,
          export_perf_results_new_api,
          false,
          "Tells to initialize new API for exporting performance metrics");

ABSL_FLAG(bool,
          webrtc_quick_perf_test,
          false,
          "Runs webrtc perfomance tests in quick mode.");
