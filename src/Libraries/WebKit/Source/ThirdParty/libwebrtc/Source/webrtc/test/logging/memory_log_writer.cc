/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#include "test/logging/memory_log_writer.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {
class MemoryLogWriter final : public RtcEventLogOutput {
 public:
  explicit MemoryLogWriter(std::map<std::string, std::string>* target,
                           absl::string_view filename)
      : target_(target), filename_(filename) {}
  ~MemoryLogWriter() final { target_->insert({filename_, std::move(buffer_)}); }
  bool IsActive() const override { return true; }
  bool Write(absl::string_view value) override {
    buffer_.append(value.data(), value.size());
    return true;
  }
  void Flush() override {}

 private:
  std::map<std::string, std::string>* const target_;
  const std::string filename_;
  std::string buffer_;
};

class MemoryLogWriterFactory final : public LogWriterFactoryInterface {
 public:
  explicit MemoryLogWriterFactory(std::map<std::string, std::string>* target)
      : target_(target) {}
  ~MemoryLogWriterFactory() override {}
  std::unique_ptr<RtcEventLogOutput> Create(
      absl::string_view filename) override {
    return std::make_unique<MemoryLogWriter>(target_, filename);
  }

 private:
  std::map<std::string, std::string>* const target_;
};

}  // namespace

MemoryLogStorage::MemoryLogStorage() {}

MemoryLogStorage::~MemoryLogStorage() {}

std::unique_ptr<LogWriterFactoryInterface> MemoryLogStorage::CreateFactory() {
  return std::make_unique<MemoryLogWriterFactory>(&logs_);
}

// namespace webrtc_impl
}  // namespace webrtc
