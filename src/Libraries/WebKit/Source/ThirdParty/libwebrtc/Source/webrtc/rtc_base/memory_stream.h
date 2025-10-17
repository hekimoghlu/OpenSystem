/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#ifndef RTC_BASE_MEMORY_STREAM_H_
#define RTC_BASE_MEMORY_STREAM_H_

#include <stddef.h>

#include "rtc_base/stream.h"

namespace rtc {

// MemoryStream dynamically resizes to accomodate written data.

class MemoryStream final : public StreamInterface {
 public:
  MemoryStream();
  ~MemoryStream() override;

  StreamState GetState() const override;
  StreamResult Read(rtc::ArrayView<uint8_t> buffer,
                    size_t& bytes_read,
                    int& error) override;
  StreamResult Write(rtc::ArrayView<const uint8_t> buffer,
                     size_t& bytes_written,
                     int& error) override;
  void Close() override;
  bool GetSize(size_t* size) const;
  bool ReserveSize(size_t size);

  bool SetPosition(size_t position);
  bool GetPosition(size_t* position) const;
  void Rewind();

  char* GetBuffer() { return buffer_; }
  const char* GetBuffer() const { return buffer_; }

  void SetData(const void* data, size_t length);

 private:
  StreamResult DoReserve(size_t size, int* error);

  // Invariant: 0 <= seek_position <= data_length_ <= buffer_length_
  char* buffer_ = nullptr;
  size_t buffer_length_ = 0;
  size_t data_length_ = 0;
  size_t seek_position_ = 0;
};

}  // namespace rtc

#endif  // RTC_BASE_MEMORY_STREAM_H_
