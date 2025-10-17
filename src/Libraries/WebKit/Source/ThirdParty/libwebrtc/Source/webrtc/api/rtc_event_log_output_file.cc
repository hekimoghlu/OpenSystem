/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#include "api/rtc_event_log_output_file.h"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/rtc_event_log/rtc_event_log.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {

// Together with the assumption of no single Write() would ever be called on
// an input with length greater-than-or-equal-to (max(size_t) / 2), this
// guarantees no overflow of the check for remaining file capacity in Write().
// This does *not* apply to files with unlimited size.
const size_t RtcEventLogOutputFile::kMaxReasonableFileSize =
    std::numeric_limits<size_t>::max() / 2;

RtcEventLogOutputFile::RtcEventLogOutputFile(const std::string& file_name)
    : RtcEventLogOutputFile(FileWrapper::OpenWriteOnly(file_name),
                            RtcEventLog::kUnlimitedOutput) {}

RtcEventLogOutputFile::RtcEventLogOutputFile(const std::string& file_name,
                                             size_t max_size_bytes)

    // Unlike plain fopen, FileWrapper takes care of filename utf8 ->
    // wchar conversion on Windows.
    : RtcEventLogOutputFile(FileWrapper::OpenWriteOnly(file_name),
                            max_size_bytes) {}

RtcEventLogOutputFile::RtcEventLogOutputFile(FILE* file, size_t max_size_bytes)
    : RtcEventLogOutputFile(FileWrapper(file), max_size_bytes) {}

RtcEventLogOutputFile::RtcEventLogOutputFile(FileWrapper file,
                                             size_t max_size_bytes)
    : max_size_bytes_(max_size_bytes), file_(std::move(file)) {
  RTC_CHECK_LE(max_size_bytes_, kMaxReasonableFileSize);
  if (!file_.is_open()) {
    RTC_LOG(LS_ERROR) << "Invalid file. WebRTC event log not started.";
  }
}

bool RtcEventLogOutputFile::IsActive() const {
  return IsActiveInternal();
}

bool RtcEventLogOutputFile::Write(absl::string_view output) {
  RTC_DCHECK(IsActiveInternal());
  // No single write may be so big, that it would risk overflowing the
  // calculation of (written_bytes_ + output.length()).
  RTC_DCHECK_LT(output.size(), kMaxReasonableFileSize);

  if (max_size_bytes_ == RtcEventLog::kUnlimitedOutput ||
      written_bytes_ + output.size() <= max_size_bytes_) {
    if (file_.Write(output.data(), output.size())) {
      written_bytes_ += output.size();
      return true;
    } else {
      RTC_LOG(LS_ERROR) << "Write to WebRtcEventLog file failed.";
    }
  } else {
    RTC_LOG(LS_VERBOSE) << "Max file size reached.";
  }

  // Failed, for one of above reasons. Close output file.
  file_.Close();
  return false;
}

// Internal non-virtual method.
bool RtcEventLogOutputFile::IsActiveInternal() const {
  return file_.is_open();
}

}  // namespace webrtc
