/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include "rtc_base/system/file_wrapper.h"

#include <stddef.h>

#include <cerrno>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

#ifdef _WIN32
#include <Windows.h>
#else
#endif

#include <utility>

namespace webrtc {
namespace {
FILE* FileOpen(absl::string_view file_name_utf8, bool read_only, int* error) {
  RTC_CHECK_EQ(file_name_utf8.find_first_of('\0'), absl::string_view::npos)
      << "Invalid filename, containing NUL character";
  std::string file_name(file_name_utf8);
#if defined(_WIN32)
  int len = MultiByteToWideChar(CP_UTF8, 0, file_name.c_str(), -1, nullptr, 0);
  std::wstring wstr(len, 0);
  MultiByteToWideChar(CP_UTF8, 0, file_name.c_str(), -1, &wstr[0], len);
  FILE* file = _wfopen(wstr.c_str(), read_only ? L"rb" : L"wb");
#else
  FILE* file = fopen(file_name.c_str(), read_only ? "rb" : "wb");
#endif
  if (!file && error) {
    *error = errno;
  }
  return file;
}

}  // namespace

// static
FileWrapper FileWrapper::OpenReadOnly(absl::string_view file_name_utf8) {
  return FileWrapper(FileOpen(file_name_utf8, true, nullptr));
}

// static
FileWrapper FileWrapper::OpenWriteOnly(absl::string_view file_name_utf8,
                                       int* error /*=nullptr*/) {
  return FileWrapper(FileOpen(file_name_utf8, false, error));
}

FileWrapper::FileWrapper(FileWrapper&& other) {
  operator=(std::move(other));
}

FileWrapper& FileWrapper::operator=(FileWrapper&& other) {
  Close();
  file_ = other.file_;
  other.file_ = nullptr;
  return *this;
}

bool FileWrapper::SeekRelative(int64_t offset) {
  RTC_DCHECK(file_);
  return fseek(file_, rtc::checked_cast<long>(offset), SEEK_CUR) == 0;
}

bool FileWrapper::SeekTo(int64_t position) {
  RTC_DCHECK(file_);
  return fseek(file_, rtc::checked_cast<long>(position), SEEK_SET) == 0;
}

std::optional<size_t> FileWrapper::FileSize() {
  if (file_ == nullptr)
    return std::nullopt;
  long original_position = ftell(file_);
  if (original_position < 0)
    return std::nullopt;
  int seek_error = fseek(file_, 0, SEEK_END);
  if (seek_error)
    return std::nullopt;
  long file_size = ftell(file_);
  seek_error = fseek(file_, original_position, SEEK_SET);
  if (seek_error)
    return std::nullopt;
  return rtc::checked_cast<size_t>(file_size);
}

bool FileWrapper::Flush() {
  RTC_DCHECK(file_);
  return fflush(file_) == 0;
}

size_t FileWrapper::Read(void* buf, size_t length) {
  RTC_DCHECK(file_);
  return fread(buf, 1, length, file_);
}

bool FileWrapper::ReadEof() const {
  RTC_DCHECK(file_);
  return feof(file_);
}

bool FileWrapper::Write(const void* buf, size_t length) {
  RTC_DCHECK(file_);
  return fwrite(buf, 1, length, file_) == length;
}

bool FileWrapper::Close() {
  if (file_ == nullptr)
    return true;

  bool success = fclose(file_) == 0;
  file_ = nullptr;
  return success;
}

FILE* FileWrapper::Release() {
  FILE* file = file_;
  file_ = nullptr;
  return file;
}

}  // namespace webrtc
