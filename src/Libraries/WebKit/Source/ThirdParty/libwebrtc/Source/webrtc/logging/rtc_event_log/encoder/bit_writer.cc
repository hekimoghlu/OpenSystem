/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "logging/rtc_event_log/encoder/bit_writer.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {

namespace {
size_t BitsToBytes(size_t bits) {
  return (bits / 8) + (bits % 8 > 0 ? 1 : 0);
}
}  // namespace

void BitWriter::WriteBits(uint64_t val, size_t bit_count) {
  RTC_DCHECK(valid_);
  const bool success = bit_writer_.WriteBits(val, bit_count);
  RTC_DCHECK(success);
  written_bits_ += bit_count;
}

void BitWriter::WriteBits(absl::string_view input) {
  RTC_DCHECK(valid_);
  for (char c : input) {
    WriteBits(static_cast<unsigned char>(c), CHAR_BIT);
  }
}

// Returns everything that was written so far.
// Nothing more may be written after this is called.
std::string BitWriter::GetString() {
  RTC_DCHECK(valid_);
  valid_ = false;

  buffer_.resize(BitsToBytes(written_bits_));
  written_bits_ = 0;

  std::string result;
  std::swap(buffer_, result);
  return result;
}

}  // namespace webrtc
