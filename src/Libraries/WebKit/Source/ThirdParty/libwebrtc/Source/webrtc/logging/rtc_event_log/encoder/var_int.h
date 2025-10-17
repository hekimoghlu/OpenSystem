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
#ifndef LOGGING_RTC_EVENT_LOG_ENCODER_VAR_INT_H_
#define LOGGING_RTC_EVENT_LOG_ENCODER_VAR_INT_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "rtc_base/bitstream_reader.h"

namespace webrtc {

extern const size_t kMaxVarIntLengthBytes;

// Encode a given uint64_t as a varint. From least to most significant,
// each batch of seven bits are put into the lower bits of a byte, and the last
// remaining bit in that byte (the highest one) marks whether additional bytes
// follow (which happens if and only if there are other bits in `input` which
// are non-zero).
// Notes: If input == 0, one byte is used. If input is uint64_t::max, exactly
// kMaxVarIntLengthBytes are used.
std::string EncodeVarInt(uint64_t input);

// Inverse of EncodeVarInt().
// Returns true and the remaining (unread) slice of the input if decoding
// succeeds. Returns false otherwise and `output` is not modified.
std::pair<bool, absl::string_view> DecodeVarInt(absl::string_view input,
                                                uint64_t* output);

// Same as other version, but uses a BitstreamReader for input.
// If decoding is successful returns the decoded varint.
// If not successful, `input` reader is set into the failure state, return value
// is unspecified.
uint64_t DecodeVarInt(BitstreamReader& input);

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ENCODER_VAR_INT_H_
