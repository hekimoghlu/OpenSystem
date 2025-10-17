/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#include "modules/rtp_rtcp/source/rtp_dependency_descriptor_extension.h"

#include <bitset>
#include <cstdint>

#include "api/array_view.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_dependency_descriptor_reader.h"
#include "modules/rtp_rtcp/source/rtp_dependency_descriptor_writer.h"
#include "rtc_base/numerics/divide_round.h"

namespace webrtc {

bool RtpDependencyDescriptorExtension::Parse(
    rtc::ArrayView<const uint8_t> data,
    const FrameDependencyStructure* structure,
    DependencyDescriptor* descriptor) {
  RtpDependencyDescriptorReader reader(data, structure, descriptor);
  return reader.ParseSuccessful();
}

size_t RtpDependencyDescriptorExtension::ValueSize(
    const FrameDependencyStructure& structure,
    std::bitset<32> active_chains,
    const DependencyDescriptor& descriptor) {
  RtpDependencyDescriptorWriter writer(/*data=*/{}, structure, active_chains,
                                       descriptor);
  return DivideRoundUp(writer.ValueSizeBits(), 8);
}

bool RtpDependencyDescriptorExtension::Write(
    rtc::ArrayView<uint8_t> data,
    const FrameDependencyStructure& structure,
    std::bitset<32> active_chains,
    const DependencyDescriptor& descriptor) {
  RtpDependencyDescriptorWriter writer(data, structure, active_chains,
                                       descriptor);
  return writer.Write();
}

bool RtpDependencyDescriptorExtension::Parse(
    rtc::ArrayView<const uint8_t> data,
    DependencyDescriptorMandatory* descriptor) {
  if (data.size() < 3) {
    return false;
  }
  descriptor->set_first_packet_in_frame(data[0] & 0b1000'0000);
  descriptor->set_last_packet_in_frame(data[0] & 0b0100'0000);
  descriptor->set_template_id(data[0] & 0b0011'1111);
  descriptor->set_frame_number((uint16_t{data[1]} << 8) | data[2]);
  return true;
}

}  // namespace webrtc
