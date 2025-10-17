/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#include "test/logging/file_log_writer.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace webrtc_impl {

FileLogWriter::FileLogWriter(absl::string_view file_path)
    : out_(std::fopen(std::string(file_path).c_str(), "wb")) {
  RTC_CHECK(out_ != nullptr)
      << "Failed to open file: '" << file_path << "' for writing.";
}

FileLogWriter::~FileLogWriter() {
  std::fclose(out_);
}

bool FileLogWriter::IsActive() const {
  return true;
}

bool FileLogWriter::Write(absl::string_view value) {
  // We don't expect the write to fail. If it does, we don't want to risk
  // silently ignoring it.
  RTC_CHECK_EQ(std::fwrite(value.data(), 1, value.size(), out_), value.size())
      << "fwrite failed unexpectedly: " << errno;
  return true;
}

void FileLogWriter::Flush() {
  RTC_CHECK_EQ(fflush(out_), 0) << "fflush failed unexpectedly: " << errno;
}

}  // namespace webrtc_impl

FileLogWriterFactory::FileLogWriterFactory(absl::string_view base_path)
    : base_path_(base_path) {
  for (size_t i = 0; i < base_path.size(); ++i) {
    if (base_path[i] == '/')
      test::CreateDir(base_path.substr(0, i));
  }
}

FileLogWriterFactory::~FileLogWriterFactory() {}

std::unique_ptr<RtcEventLogOutput> FileLogWriterFactory::Create(
    absl::string_view filename) {
  return std::make_unique<webrtc_impl::FileLogWriter>(base_path_ +
                                                      std::string(filename));
}
}  // namespace webrtc
