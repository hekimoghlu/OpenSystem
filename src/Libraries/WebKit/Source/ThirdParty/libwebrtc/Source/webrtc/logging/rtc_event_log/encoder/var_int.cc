/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#include "logging/rtc_event_log/encoder/var_int.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "rtc_base/bitstream_reader.h"
#include "rtc_base/checks.h"

// TODO(eladalon): Add unit tests.

namespace webrtc {

const size_t kMaxVarIntLengthBytes = 10;  // ceil(64 / 7.0) is 10.

std::string EncodeVarInt(uint64_t input) {
  std::string output;
  output.reserve(kMaxVarIntLengthBytes);

  do {
    uint8_t byte = static_cast<uint8_t>(input & 0x7f);
    input >>= 7;
    if (input > 0) {
      byte |= 0x80;
    }
    output += byte;
  } while (input > 0);

  RTC_DCHECK_GE(output.size(), 1u);
  RTC_DCHECK_LE(output.size(), kMaxVarIntLengthBytes);

  return output;
}

// There is some code duplication between the flavors of this function.
// For performance's sake, it's best to just keep it.
std::pair<bool, absl::string_view> DecodeVarInt(absl::string_view input,
                                                uint64_t* output) {
  RTC_DCHECK(output);

  uint64_t decoded = 0;
  for (size_t i = 0; i < input.length() && i < kMaxVarIntLengthBytes; ++i) {
    decoded += (static_cast<uint64_t>(input[i] & 0x7f)
                << static_cast<uint64_t>(7 * i));
    if (!(input[i] & 0x80)) {
      *output = decoded;
      return {true, input.substr(i + 1)};
    }
  }

  return {false, input};
}

// There is some code duplication between the flavors of this function.
// For performance's sake, it's best to just keep it.
uint64_t DecodeVarInt(BitstreamReader& input) {
  uint64_t decoded = 0;
  for (size_t i = 0; i < kMaxVarIntLengthBytes; ++i) {
    uint8_t byte = input.Read<uint8_t>();
    decoded +=
        (static_cast<uint64_t>(byte & 0x7f) << static_cast<uint64_t>(7 * i));
    if (!(byte & 0x80)) {
      return decoded;
    }
  }

  input.Invalidate();
  return 0;
}

}  // namespace webrtc
