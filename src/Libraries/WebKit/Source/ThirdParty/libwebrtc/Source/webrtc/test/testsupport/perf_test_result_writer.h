/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#ifndef TEST_TESTSUPPORT_PERF_TEST_RESULT_WRITER_H_
#define TEST_TESTSUPPORT_PERF_TEST_RESULT_WRITER_H_

#include <stdio.h>

#include <string>

#include "absl/strings/string_view.h"
#include "test/testsupport/perf_test.h"

namespace webrtc {
namespace test {

// Interface for classes that write perf results to some kind of JSON format.
class PerfTestResultWriter {
 public:
  virtual ~PerfTestResultWriter() = default;

  virtual void ClearResults() = 0;
  virtual void LogResult(absl::string_view graph_name,
                         absl::string_view trace_name,
                         double value,
                         absl::string_view units,
                         bool important,
                         webrtc::test::ImproveDirection improve_direction) = 0;
  virtual void LogResultMeanAndError(
      absl::string_view graph_name,
      absl::string_view trace_name,
      double mean,
      double error,
      absl::string_view units,
      bool important,
      webrtc::test::ImproveDirection improve_direction) = 0;
  virtual void LogResultList(
      absl::string_view graph_name,
      absl::string_view trace_name,
      rtc::ArrayView<const double> values,
      absl::string_view units,
      bool important,
      webrtc::test::ImproveDirection improve_direction) = 0;

  virtual std::string Serialize() const = 0;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_TESTSUPPORT_PERF_TEST_RESULT_WRITER_H_
