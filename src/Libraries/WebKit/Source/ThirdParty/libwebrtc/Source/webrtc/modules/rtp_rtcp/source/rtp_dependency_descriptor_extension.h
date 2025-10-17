/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_EXTENSION_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_EXTENSION_H_

#include <bitset>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtp_parameters.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {
// Trait to read/write the dependency descriptor extension as described in
// https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension
class RtpDependencyDescriptorExtension {
 public:
  static constexpr RTPExtensionType kId = kRtpExtensionDependencyDescriptor;
  static constexpr absl::string_view Uri() {
    return RtpExtension::kDependencyDescriptorUri;
  }

  static bool Parse(rtc::ArrayView<const uint8_t> data,
                    const FrameDependencyStructure* structure,
                    DependencyDescriptor* descriptor);

  // Reads the mandatory part of the descriptor.
  // Such read is stateless, i.e., doesn't require `FrameDependencyStructure`.
  static bool Parse(rtc::ArrayView<const uint8_t> data,
                    DependencyDescriptorMandatory* descriptor);

  static size_t ValueSize(const FrameDependencyStructure& structure,
                          const DependencyDescriptor& descriptor) {
    return ValueSize(structure, kAllChainsAreActive, descriptor);
  }
  static size_t ValueSize(const FrameDependencyStructure& structure,
                          std::bitset<32> active_chains,
                          const DependencyDescriptor& descriptor);
  static bool Write(rtc::ArrayView<uint8_t> data,
                    const FrameDependencyStructure& structure,
                    const DependencyDescriptor& descriptor) {
    return Write(data, structure, kAllChainsAreActive, descriptor);
  }
  static bool Write(rtc::ArrayView<uint8_t> data,
                    const FrameDependencyStructure& structure,
                    std::bitset<32> active_chains,
                    const DependencyDescriptor& descriptor);

 private:
  static constexpr std::bitset<32> kAllChainsAreActive = ~uint32_t{0};
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_DEPENDENCY_DESCRIPTOR_EXTENSION_H_
