/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#ifndef RTC_BASE_LOG_SINKS_H_
#define RTC_BASE_LOG_SINKS_H_

#include <stddef.h>

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/file_rotating_stream.h"
#include "rtc_base/logging.h"

namespace rtc {

// Log sink that uses a FileRotatingStream to write to disk.
// Init() must be called before adding this sink.
class FileRotatingLogSink : public LogSink {
 public:
  // `num_log_files` must be greater than 1 and `max_log_size` must be greater
  // than 0.
  FileRotatingLogSink(absl::string_view log_dir_path,
                      absl::string_view log_prefix,
                      size_t max_log_size,
                      size_t num_log_files);
  ~FileRotatingLogSink() override;

  FileRotatingLogSink(const FileRotatingLogSink&) = delete;
  FileRotatingLogSink& operator=(const FileRotatingLogSink&) = delete;

  // Writes the message to the current file. It will spill over to the next
  // file if needed.
  void OnLogMessage(const std::string& message) override;
  void OnLogMessage(absl::string_view message) override;
  void OnLogMessage(const std::string& message,
                    LoggingSeverity sev,
                    const char* tag) override;
  void OnLogMessage(absl::string_view message,
                    LoggingSeverity sev,
                    const char* tag) override;

  // Deletes any existing files in the directory and creates a new log file.
  virtual bool Init();

  // Disables buffering on the underlying stream.
  bool DisableBuffering();

 protected:
  explicit FileRotatingLogSink(FileRotatingStream* stream);

 private:
  std::unique_ptr<FileRotatingStream> stream_;
};

// Log sink that uses a CallSessionFileRotatingStream to write to disk.
// Init() must be called before adding this sink.
class CallSessionFileRotatingLogSink : public FileRotatingLogSink {
 public:
  CallSessionFileRotatingLogSink(absl::string_view log_dir_path,
                                 size_t max_total_log_size);
  ~CallSessionFileRotatingLogSink() override;

  CallSessionFileRotatingLogSink(const CallSessionFileRotatingLogSink&) =
      delete;
  CallSessionFileRotatingLogSink& operator=(
      const CallSessionFileRotatingLogSink&) = delete;
};

}  // namespace rtc

#endif  // RTC_BASE_LOG_SINKS_H_
