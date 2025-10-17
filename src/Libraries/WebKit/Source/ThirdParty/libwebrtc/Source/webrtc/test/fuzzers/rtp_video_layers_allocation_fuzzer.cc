/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#include <cstddef>
#include <cstdint>
#include <limits>

#include "api/array_view.h"
#include "api/video/video_layers_allocation.h"
#include "modules/rtp_rtcp/source/rtp_video_layers_allocation_extension.h"
#include "rtc_base/checks.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  // Video layers allocation is an rtp header extension.
  // Per https://datatracker.ietf.org/doc/html/rfc8285#section-4.3
  // rtp header extension uses up to one byte to store the size, i.e.
  // maximum size of any rtp header extension is 255 bytes.
  constexpr int kMaxSize = std::numeric_limits<uint8_t>::max();
  if (size > kMaxSize) {
    return;
  }
  auto raw = rtc::MakeArrayView(data, size);

  VideoLayersAllocation allocation1;
  if (!RtpVideoLayersAllocationExtension::Parse(raw, &allocation1)) {
    // Ignore invalid buffer and move on.
    return;
  }

  // Write parsed allocation back into raw buffer.
  size_t value_size = RtpVideoLayersAllocationExtension::ValueSize(allocation1);
  // Check `writer` use minimal number of bytes to pack the extension by
  // checking it doesn't use more than reader consumed.
  RTC_CHECK_LE(value_size, raw.size());
  uint8_t some_memory[kMaxSize];
  RTC_CHECK_LE(value_size, kMaxSize);
  rtc::ArrayView<uint8_t> write_buffer(some_memory, value_size);
  RTC_CHECK(
      RtpVideoLayersAllocationExtension::Write(write_buffer, allocation1));

  // Parse what Write assembled.
  // Unlike random input that should always succeed.
  VideoLayersAllocation allocation2;
  RTC_CHECK(
      RtpVideoLayersAllocationExtension::Parse(write_buffer, &allocation2));

  RTC_CHECK_EQ(allocation1.rtp_stream_index, allocation2.rtp_stream_index);
  RTC_CHECK_EQ(allocation1.resolution_and_frame_rate_is_valid,
               allocation2.resolution_and_frame_rate_is_valid);
  RTC_CHECK_EQ(allocation1.active_spatial_layers.size(),
               allocation2.active_spatial_layers.size());
  RTC_CHECK(allocation1 == allocation2);
}

}  // namespace webrtc
