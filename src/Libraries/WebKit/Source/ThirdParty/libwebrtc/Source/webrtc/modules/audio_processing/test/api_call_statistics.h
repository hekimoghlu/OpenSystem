/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_TEST_API_CALL_STATISTICS_H_
#define MODULES_AUDIO_PROCESSING_TEST_API_CALL_STATISTICS_H_

#include <vector>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {

// Collects statistics about the API call durations.
class ApiCallStatistics {
 public:
  enum class CallType { kRender, kCapture };

  // Adds a new datapoint.
  void Add(int64_t duration_nanos, CallType call_type);

  // Prints out a report of the statistics.
  void PrintReport() const;

  // Writes the call information to a file.
  void WriteReportToFile(absl::string_view filename) const;

 private:
  struct CallData {
    CallData(int64_t duration_nanos, CallType call_type);
    int64_t duration_nanos;
    CallType call_type;
  };
  std::vector<CallData> calls_;
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_TEST_API_CALL_STATISTICS_H_
