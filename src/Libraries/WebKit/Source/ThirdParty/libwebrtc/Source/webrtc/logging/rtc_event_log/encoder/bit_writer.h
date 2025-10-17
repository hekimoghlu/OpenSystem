/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
#ifndef LOGGING_RTC_EVENT_LOG_ENCODER_BIT_WRITER_H_
#define LOGGING_RTC_EVENT_LOG_ENCODER_BIT_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/bit_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Wrap BitBufferWriter and extend its functionality by (1) keeping track of
// the number of bits written and (2) owning its buffer.
class BitWriter final {
 public:
  explicit BitWriter(size_t byte_count)
      : buffer_(byte_count, '\0'),
        bit_writer_(reinterpret_cast<uint8_t*>(&buffer_[0]), buffer_.size()),
        written_bits_(0),
        valid_(true) {
    RTC_DCHECK_GT(byte_count, 0);
  }

  BitWriter(const BitWriter&) = delete;
  BitWriter& operator=(const BitWriter&) = delete;

  void WriteBits(uint64_t val, size_t bit_count);

  void WriteBits(absl::string_view input);

  // Returns everything that was written so far.
  // Nothing more may be written after this is called.
  std::string GetString();

 private:
  std::string buffer_;
  rtc::BitBufferWriter bit_writer_;
  // Note: Counting bits instead of bytes wraps around earlier than it has to,
  // which means the maximum length is lower than it could be. We don't expect
  // to go anywhere near the limit, though, so this is good enough.
  size_t written_bits_;
  bool valid_;
};

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ENCODER_BIT_WRITER_H_
