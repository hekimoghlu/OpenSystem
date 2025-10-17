/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TEST_RESULT_SINK_H_
#define MODULES_AUDIO_CODING_NETEQ_TEST_RESULT_SINK_H_

#include <cstdio>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/neteq/neteq.h"
#include "rtc_base/message_digest.h"

namespace webrtc {

class ResultSink {
 public:
  explicit ResultSink(absl::string_view output_file);
  ~ResultSink();

  template <typename T>
  void AddResult(const T* test_results, size_t length);

  void AddResult(const NetEqNetworkStatistics& stats);

  void VerifyChecksum(absl::string_view ref_check_sum);

 private:
  FILE* output_fp_;
  std::unique_ptr<rtc::MessageDigest> digest_;
};

template <typename T>
void ResultSink::AddResult(const T* test_results, size_t length) {
  if (output_fp_) {
    ASSERT_EQ(length, fwrite(test_results, sizeof(T), length, output_fp_));
  }
  digest_->Update(test_results, sizeof(T) * length);
}

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TEST_RESULT_SINK_H_
