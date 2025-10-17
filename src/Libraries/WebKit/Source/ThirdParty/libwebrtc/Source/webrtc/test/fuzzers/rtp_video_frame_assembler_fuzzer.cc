/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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

#include "api/video/rtp_video_frame_assembler.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "modules/rtp_rtcp/source/rtp_dependency_descriptor_extension.h"
#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor_extension.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"

namespace webrtc {

void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size == 0) {
    return;
  }
  RtpHeaderExtensionMap extensions;
  extensions.Register<RtpDependencyDescriptorExtension>(1);
  extensions.Register<RtpGenericFrameDescriptorExtension00>(2);
  RtpPacketReceived rtp_packet(&extensions);

  RtpVideoFrameAssembler assembler(
      static_cast<RtpVideoFrameAssembler::PayloadFormat>(data[0] % 6));

  for (size_t i = 1; i < size;) {
    size_t packet_size = std::min<size_t>(size - i, 300);
    if (rtp_packet.Parse(data + i, packet_size)) {
      assembler.InsertPacket(rtp_packet);
    }
    i += packet_size;
  }
}

}  // namespace webrtc
