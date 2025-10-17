/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#ifndef API_RTC_EVENT_LOG_OUTPUT_FILE_H_
#define API_RTC_EVENT_LOG_OUTPUT_FILE_H_

#include <stddef.h>
#include <stdio.h>

#include <string>

#include "absl/strings/string_view.h"
#include "api/rtc_event_log_output.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {

class RtcEventLogOutputFile final : public RtcEventLogOutput {
 public:
  static const size_t kMaxReasonableFileSize;  // Explanation at declaration.

  // Unlimited/limited-size output file (by filename).
  explicit RtcEventLogOutputFile(const std::string& file_name);
  RtcEventLogOutputFile(const std::string& file_name, size_t max_size_bytes);

  // Limited-size output file (by FILE*). This class takes ownership
  // of the FILE*, and closes it on destruction.
  RtcEventLogOutputFile(FILE* file, size_t max_size_bytes);

  ~RtcEventLogOutputFile() override = default;

  bool IsActive() const override;

  bool Write(absl::string_view output) override;

 private:
  RtcEventLogOutputFile(FileWrapper file, size_t max_size_bytes);

  // IsActive() can be called either from outside or from inside, but we don't
  // want to incur the overhead of a virtual function call if called from inside
  // some other function of this class.
  inline bool IsActiveInternal() const;

  // Maximum size, or zero for no limit.
  const size_t max_size_bytes_;
  size_t written_bytes_{0};
  FileWrapper file_;
};

}  // namespace webrtc

#endif  // API_RTC_EVENT_LOG_OUTPUT_FILE_H_
