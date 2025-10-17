/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "api/array_view.h"
#include "common_video/generic_frame_descriptor/generic_frame_info.h"
#include "modules/rtp_rtcp/source/rtp_dependency_descriptor_extension.h"
#include "rtc_base/checks.h"
#include "test/fuzzers/fuzz_data_helper.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  FrameDependencyStructure structure1;
  // nullptr during 1st while loop, after that should point to structure1.
  const FrameDependencyStructure* structure1_ptr = nullptr;
  std::unique_ptr<const FrameDependencyStructure> structure2;

  test::FuzzDataHelper fuzz_data(rtc::MakeArrayView(data, size));
  while (fuzz_data.CanReadBytes(1)) {
    // Treat next byte as size of the next extension. That aligns how
    // two-byte rtp header extension sizes are written.
    size_t next_size = fuzz_data.Read<uint8_t>();
    auto raw =
        fuzz_data.ReadByteArray(std::min(next_size, fuzz_data.BytesLeft()));

    // Read the random input.
    DependencyDescriptor descriptor1;
    if (!RtpDependencyDescriptorExtension::Parse(raw, structure1_ptr,
                                                 &descriptor1)) {
      // Ignore invalid buffer and move on.
      continue;
    }
    if (descriptor1.attached_structure) {
      structure1 = *descriptor1.attached_structure;
      structure1_ptr = &structure1;
    }
    RTC_CHECK(structure1_ptr);

    // Write parsed descriptor back into raw buffer.
    size_t value_size =
        RtpDependencyDescriptorExtension::ValueSize(structure1, descriptor1);
    // Check `writer` use minimal number of bytes to pack the descriptor by
    // checking it doesn't use more than reader consumed.
    RTC_CHECK_LE(value_size, raw.size());
    uint8_t some_memory[256];
    // That should be true because value_size <= next_size < 256
    RTC_CHECK_LT(value_size, 256);
    rtc::ArrayView<uint8_t> write_buffer(some_memory, value_size);
    RTC_CHECK(RtpDependencyDescriptorExtension::Write(write_buffer, structure1,
                                                      descriptor1));

    // Parse what Write assembled.
    // Unlike random input that should always succeed.
    DependencyDescriptor descriptor2;
    RTC_CHECK(RtpDependencyDescriptorExtension::Parse(
        write_buffer, structure2.get(), &descriptor2));
    // Check descriptor1 and descriptor2 have same values.
    RTC_CHECK_EQ(descriptor1.first_packet_in_frame,
                 descriptor2.first_packet_in_frame);
    RTC_CHECK_EQ(descriptor1.last_packet_in_frame,
                 descriptor2.last_packet_in_frame);
    RTC_CHECK_EQ(descriptor1.attached_structure != nullptr,
                 descriptor2.attached_structure != nullptr);
    // Using value_or would miss invalid corner case when one value is nullopt
    // while another one is 0, but for other errors would produce much nicer
    // error message than using RTC_CHECK(optional1 == optional2);
    // If logger would support pretty printing optional values, value_or can be
    // removed.
    RTC_CHECK_EQ(descriptor1.active_decode_targets_bitmask.value_or(0),
                 descriptor2.active_decode_targets_bitmask.value_or(0));
    RTC_CHECK_EQ(descriptor1.frame_number, descriptor2.frame_number);
    RTC_CHECK(descriptor1.resolution == descriptor2.resolution);
    RTC_CHECK(descriptor1.frame_dependencies == descriptor2.frame_dependencies);

    if (descriptor2.attached_structure) {
      structure2 = std::move(descriptor2.attached_structure);
    }
  }
}

}  // namespace webrtc
