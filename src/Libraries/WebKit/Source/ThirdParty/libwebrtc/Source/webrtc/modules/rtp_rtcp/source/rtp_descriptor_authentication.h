/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_DESCRIPTOR_AUTHENTICATION_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_DESCRIPTOR_AUTHENTICATION_H_

#include <cstdint>
#include <vector>

#include "modules/rtp_rtcp/source/rtp_video_header.h"

namespace webrtc {

// Converts frame dependencies into array of bytes for authentication.
std::vector<uint8_t> RtpDescriptorAuthentication(
    const RTPVideoHeader& rtp_video_header);

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_DESCRIPTOR_AUTHENTICATION_H_
