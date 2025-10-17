/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#include "common_video/h265/h265_common.h"

#include "common_video/h264/h264_common.h"

namespace webrtc {
namespace H265 {

constexpr uint8_t kNaluTypeMask = 0x7E;

std::vector<NaluIndex> FindNaluIndices(rtc::ArrayView<const uint8_t> buffer) {
  std::vector<H264::NaluIndex> indices = H264::FindNaluIndices(buffer);
  std::vector<NaluIndex> results;
  for (auto& index : indices) {
    results.push_back(
        {index.start_offset, index.payload_start_offset, index.payload_size});
  }
  return results;
}

NaluType ParseNaluType(uint8_t data) {
  return static_cast<NaluType>((data & kNaluTypeMask) >> 1);
}

std::vector<uint8_t> ParseRbsp(rtc::ArrayView<const uint8_t> data) {
  return H264::ParseRbsp(data);
}

void WriteRbsp(rtc::ArrayView<const uint8_t> bytes, rtc::Buffer* destination) {
  H264::WriteRbsp(bytes, destination);
}

uint32_t Log2Ceiling(uint32_t value) {
  // When n == 0, we want the function to return -1.
  // When n == 0, (n - 1) will underflow to 0xFFFFFFFF, which is
  // why the statement below starts with (n ? 32 : -1).
  return (value ? 32 : -1) - WebRtcVideo_CountLeadingZeros32(value - 1);
}

}  // namespace H265
}  // namespace webrtc
