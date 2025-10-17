/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_VIDEO_LAYERS_ALLOCATION_EXTENSION_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_VIDEO_LAYERS_ALLOCATION_EXTENSION_H_

#include "absl/strings/string_view.h"
#include "api/rtp_parameters.h"
#include "api/video/video_layers_allocation.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {

class RtpVideoLayersAllocationExtension {
 public:
  using value_type = VideoLayersAllocation;
  static constexpr RTPExtensionType kId = kRtpExtensionVideoLayersAllocation;
  static constexpr absl::string_view Uri() {
    return RtpExtension::kVideoLayersAllocationUri;
  }

  static bool Parse(rtc::ArrayView<const uint8_t> data,
                    VideoLayersAllocation* allocation);
  static size_t ValueSize(const VideoLayersAllocation& allocation);
  static bool Write(rtc::ArrayView<uint8_t> data,
                    const VideoLayersAllocation& allocation);
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_VIDEO_LAYERS_ALLOCATION_EXTENSION_H_
