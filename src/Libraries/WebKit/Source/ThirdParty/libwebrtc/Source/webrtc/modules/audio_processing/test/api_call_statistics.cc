/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
#include "modules/audio_processing/test/api_call_statistics.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/time_utils.h"

namespace webrtc {
namespace test {

void ApiCallStatistics::Add(int64_t duration_nanos, CallType call_type) {
  calls_.push_back(CallData(duration_nanos, call_type));
}

void ApiCallStatistics::PrintReport() const {
  int64_t min_render = std::numeric_limits<int64_t>::max();
  int64_t min_capture = std::numeric_limits<int64_t>::max();
  int64_t max_render = 0;
  int64_t max_capture = 0;
  int64_t sum_render = 0;
  int64_t sum_capture = 0;
  int64_t num_render = 0;
  int64_t num_capture = 0;
  int64_t avg_render = 0;
  int64_t avg_capture = 0;

  for (auto v : calls_) {
    if (v.call_type == CallType::kRender) {
      ++num_render;
      min_render = std::min(min_render, v.duration_nanos);
      max_render = std::max(max_render, v.duration_nanos);
      sum_render += v.duration_nanos;
    } else {
      ++num_capture;
      min_capture = std::min(min_capture, v.duration_nanos);
      max_capture = std::max(max_capture, v.duration_nanos);
      sum_capture += v.duration_nanos;
    }
  }
  min_render /= rtc::kNumNanosecsPerMicrosec;
  max_render /= rtc::kNumNanosecsPerMicrosec;
  sum_render /= rtc::kNumNanosecsPerMicrosec;
  min_capture /= rtc::kNumNanosecsPerMicrosec;
  max_capture /= rtc::kNumNanosecsPerMicrosec;
  sum_capture /= rtc::kNumNanosecsPerMicrosec;
  avg_render = num_render > 0 ? sum_render / num_render : 0;
  avg_capture = num_capture > 0 ? sum_capture / num_capture : 0;

  std::cout << std::endl
            << "Total time: " << (sum_capture + sum_render) * 1e-6 << " s"
            << std::endl
            << " Render API calls:" << std::endl
            << "   min: " << min_render << " us" << std::endl
            << "   max: " << max_render << " us" << std::endl
            << "   avg: " << avg_render << " us" << std::endl
            << " Capture API calls:" << std::endl
            << "   min: " << min_capture << " us" << std::endl
            << "   max: " << max_capture << " us" << std::endl
            << "   avg: " << avg_capture << " us" << std::endl;
}

void ApiCallStatistics::WriteReportToFile(absl::string_view filename) const {
  std::unique_ptr<std::ofstream> out =
      std::make_unique<std::ofstream>(std::string(filename));
  for (auto v : calls_) {
    if (v.call_type == CallType::kRender) {
      *out << "render, ";
    } else {
      *out << "capture, ";
    }
    *out << (v.duration_nanos / rtc::kNumNanosecsPerMicrosec) << std::endl;
  }
}

ApiCallStatistics::CallData::CallData(int64_t duration_nanos,
                                      CallType call_type)
    : duration_nanos(duration_nanos), call_type(call_type) {}

}  // namespace test
}  // namespace webrtc
