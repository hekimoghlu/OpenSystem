/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_GENERIC_FRAME_DESCRIPTOR_EXTENSION_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_GENERIC_FRAME_DESCRIPTOR_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "api/rtp_parameters.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"
#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor.h"

namespace webrtc {

// Trait to read/write the generic frame descriptor, the early version of the
// dependency descriptor extension. Usage of this rtp header extension is
// discouraged in favor of the dependency descriptor.
class RtpGenericFrameDescriptorExtension00 {
 public:
  using value_type = RtpGenericFrameDescriptor;
  static constexpr RTPExtensionType kId = kRtpExtensionGenericFrameDescriptor;
  static constexpr absl::string_view Uri() {
    return RtpExtension::kGenericFrameDescriptorUri00;
  }
  static constexpr int kMaxSizeBytes = 16;

  static bool Parse(rtc::ArrayView<const uint8_t> data,
                    RtpGenericFrameDescriptor* descriptor);
  static size_t ValueSize(const RtpGenericFrameDescriptor& descriptor);
  static bool Write(rtc::ArrayView<uint8_t> data,
                    const RtpGenericFrameDescriptor& descriptor);
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_GENERIC_FRAME_DESCRIPTOR_EXTENSION_H_
