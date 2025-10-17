/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#include "rtc_base/log_sinks.h"

#include <string.h>

#include <cstdio>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace rtc {

FileRotatingLogSink::FileRotatingLogSink(absl::string_view log_dir_path,
                                         absl::string_view log_prefix,
                                         size_t max_log_size,
                                         size_t num_log_files)
    : FileRotatingLogSink(new FileRotatingStream(log_dir_path,
                                                 log_prefix,
                                                 max_log_size,
                                                 num_log_files)) {}

FileRotatingLogSink::FileRotatingLogSink(FileRotatingStream* stream)
    : stream_(stream) {
  RTC_DCHECK(stream);
}

FileRotatingLogSink::~FileRotatingLogSink() {}

void FileRotatingLogSink::OnLogMessage(const std::string& message) {
  OnLogMessage(absl::string_view(message));
}

void FileRotatingLogSink::OnLogMessage(absl::string_view message) {
  if (!stream_->IsOpen()) {
    std::fprintf(stderr, "Init() must be called before adding this sink.\n");
    return;
  }
  stream_->Write(message.data(), message.size());
}

void FileRotatingLogSink::OnLogMessage(const std::string& message,
                                       LoggingSeverity sev,
                                       const char* tag) {
  OnLogMessage(absl::string_view(message), sev, tag);
}

void FileRotatingLogSink::OnLogMessage(absl::string_view message,
                                       LoggingSeverity sev,
                                       const char* tag) {
  if (!stream_->IsOpen()) {
    std::fprintf(stderr, "Init() must be called before adding this sink.\n");
    return;
  }
  stream_->Write(tag, strlen(tag));
  stream_->Write(": ", 2);
  stream_->Write(message.data(), message.size());
}

bool FileRotatingLogSink::Init() {
  return stream_->Open();
}

bool FileRotatingLogSink::DisableBuffering() {
  return stream_->DisableBuffering();
}

CallSessionFileRotatingLogSink::CallSessionFileRotatingLogSink(
    absl::string_view log_dir_path,
    size_t max_total_log_size)
    : FileRotatingLogSink(
          new CallSessionFileRotatingStream(log_dir_path, max_total_log_size)) {
}

CallSessionFileRotatingLogSink::~CallSessionFileRotatingLogSink() {}

}  // namespace rtc
