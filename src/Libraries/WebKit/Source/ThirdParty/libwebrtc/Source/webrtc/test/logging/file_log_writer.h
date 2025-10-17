/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#ifndef TEST_LOGGING_FILE_LOG_WRITER_H_
#define TEST_LOGGING_FILE_LOG_WRITER_H_

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "test/logging/log_writer.h"

namespace webrtc {
namespace webrtc_impl {
class FileLogWriter final : public RtcEventLogOutput {
 public:
  explicit FileLogWriter(absl::string_view file_path);
  ~FileLogWriter() final;
  bool IsActive() const override;
  bool Write(absl::string_view value) override;
  void Flush() override;

 private:
  std::FILE* const out_;
};
}  // namespace webrtc_impl
class FileLogWriterFactory final : public LogWriterFactoryInterface {
 public:
  explicit FileLogWriterFactory(absl::string_view base_path);
  ~FileLogWriterFactory() final;

  std::unique_ptr<RtcEventLogOutput> Create(
      absl::string_view filename) override;

 private:
  const std::string base_path_;
  std::vector<std::unique_ptr<webrtc_impl::FileLogWriter>> writers_;
};

}  // namespace webrtc

#endif  // TEST_LOGGING_FILE_LOG_WRITER_H_
