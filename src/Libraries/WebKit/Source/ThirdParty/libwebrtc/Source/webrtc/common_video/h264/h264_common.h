/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#ifndef COMMON_VIDEO_H264_H264_COMMON_H_
#define COMMON_VIDEO_H264_H264_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "rtc_base/buffer.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

namespace H264 {
// The size of a full NALU start sequence {0 0 0 1}, used for the first NALU
// of an access unit, and for SPS and PPS blocks.
const size_t kNaluLongStartSequenceSize = 4;

// The size of a shortened NALU start sequence {0 0 1}, that may be used if
// not the first NALU of an access unit or an SPS or PPS block.
const size_t kNaluShortStartSequenceSize = 3;

// The size of the NALU type byte (1).
const size_t kNaluTypeSize = 1;

// Maximum reference index for reference pictures.
constexpr int kMaxReferenceIndex = 31;

enum NaluType : uint8_t {
  kSlice = 1,
  kIdr = 5,
  kSei = 6,
  kSps = 7,
  kPps = 8,
  kAud = 9,
  kEndOfSequence = 10,
  kEndOfStream = 11,
  kFiller = 12,
  kPrefix = 14,
  kStapA = 24,
  kFuA = 28
};

enum SliceType : uint8_t { kP = 0, kB = 1, kI = 2, kSp = 3, kSi = 4 };

struct NaluIndex {
  // Start index of NALU, including start sequence.
  size_t start_offset;
  // Start index of NALU payload, typically type header.
  size_t payload_start_offset;
  // Length of NALU payload, in bytes, counting from payload_start_offset.
  size_t payload_size;
};

// Returns a vector of the NALU indices in the given buffer.
RTC_EXPORT std::vector<NaluIndex> FindNaluIndices(
    rtc::ArrayView<const uint8_t> buffer);

// Get the NAL type from the header byte immediately following start sequence.
RTC_EXPORT NaluType ParseNaluType(uint8_t data);

// Methods for parsing and writing RBSP. See section 7.4.1 of the H264 spec.
//
// The following sequences are illegal, and need to be escaped when encoding:
// 00 00 00 -> 00 00 03 00
// 00 00 01 -> 00 00 03 01
// 00 00 02 -> 00 00 03 02
// And things in the source that look like the emulation byte pattern (00 00 03)
// need to have an extra emulation byte added, so it's removed when decoding:
// 00 00 03 -> 00 00 03 03
//
// Decoding is simply a matter of finding any 00 00 03 sequence and removing
// the 03 emulation byte.

// Parse the given data and remove any emulation byte escaping.
std::vector<uint8_t> ParseRbsp(rtc::ArrayView<const uint8_t> data);

// TODO: bugs.webrtc.org/42225170 - Deprecate.
inline std::vector<uint8_t> ParseRbsp(const uint8_t* data, size_t length) {
  return ParseRbsp(rtc::MakeArrayView(data, length));
}

// Write the given data to the destination buffer, inserting and emulation
// bytes in order to escape any data the could be interpreted as a start
// sequence.
void WriteRbsp(rtc::ArrayView<const uint8_t> bytes, rtc::Buffer* destination);

// TODO: bugs.webrtc.org/42225170 - Deprecate.
inline void WriteRbsp(const uint8_t* bytes,
                      size_t length,
                      rtc::Buffer* destination) {
  WriteRbsp(rtc::MakeArrayView(bytes, length), destination);
}
}  // namespace H264
}  // namespace webrtc

#endif  // COMMON_VIDEO_H264_H264_COMMON_H_
