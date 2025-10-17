/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
#include "rtc_base/memory_stream.h"

#include <errno.h>
#include <string.h>

#include <algorithm>

#include "rtc_base/checks.h"

namespace rtc {

StreamState MemoryStream::GetState() const {
  return SS_OPEN;
}

StreamResult MemoryStream::Read(rtc::ArrayView<uint8_t> buffer,
                                size_t& bytes_read,
                                int& error) {
  if (seek_position_ >= data_length_) {
    return SR_EOS;
  }
  size_t available = data_length_ - seek_position_;
  size_t bytes;
  if (buffer.size() > available) {
    // Read partial buffer
    bytes = available;
  } else {
    bytes = buffer.size();
  }
  memcpy(buffer.data(), &buffer_[seek_position_], bytes);
  seek_position_ += bytes;
  bytes_read = bytes;
  return SR_SUCCESS;
}

StreamResult MemoryStream::Write(rtc::ArrayView<const uint8_t> buffer,
                                 size_t& bytes_written,
                                 int& error) {
  size_t available = buffer_length_ - seek_position_;
  if (0 == available) {
    // Increase buffer size to the larger of:
    // a) new position rounded up to next 256 bytes
    // b) double the previous length
    size_t new_buffer_length = std::max(
        ((seek_position_ + buffer.size()) | 0xFF) + 1, buffer_length_ * 2);
    StreamResult result = DoReserve(new_buffer_length, &error);
    if (SR_SUCCESS != result) {
      return result;
    }
    RTC_DCHECK(buffer_length_ >= new_buffer_length);
    available = buffer_length_ - seek_position_;
  }

  size_t bytes = buffer.size();
  if (bytes > available) {
    bytes = available;
  }
  memcpy(&buffer_[seek_position_], buffer.data(), bytes);
  seek_position_ += bytes;
  if (data_length_ < seek_position_) {
    data_length_ = seek_position_;
  }
  bytes_written = bytes;
  return SR_SUCCESS;
}

void MemoryStream::Close() {
  // nothing to do
}

bool MemoryStream::SetPosition(size_t position) {
  if (position > data_length_)
    return false;
  seek_position_ = position;
  return true;
}

bool MemoryStream::GetPosition(size_t* position) const {
  if (position)
    *position = seek_position_;
  return true;
}

void MemoryStream::Rewind() {
  seek_position_ = 0;
}

bool MemoryStream::GetSize(size_t* size) const {
  if (size)
    *size = data_length_;
  return true;
}

bool MemoryStream::ReserveSize(size_t size) {
  return (SR_SUCCESS == DoReserve(size, nullptr));
}

///////////////////////////////////////////////////////////////////////////////

MemoryStream::MemoryStream() {}

MemoryStream::~MemoryStream() {
  delete[] buffer_;
}

void MemoryStream::SetData(const void* data, size_t length) {
  data_length_ = buffer_length_ = length;
  delete[] buffer_;
  buffer_ = new char[buffer_length_];
  memcpy(buffer_, data, data_length_);
  seek_position_ = 0;
}

StreamResult MemoryStream::DoReserve(size_t size, int* error) {
  if (buffer_length_ >= size)
    return SR_SUCCESS;

  if (char* new_buffer = new char[size]) {
    if (buffer_ != nullptr && data_length_ > 0) {
      memcpy(new_buffer, buffer_, data_length_);
    }
    delete[] buffer_;
    buffer_ = new_buffer;
    buffer_length_ = size;
    return SR_SUCCESS;
  }

  if (error) {
    *error = ENOMEM;
  }
  return SR_ERROR;
}

}  // namespace rtc
