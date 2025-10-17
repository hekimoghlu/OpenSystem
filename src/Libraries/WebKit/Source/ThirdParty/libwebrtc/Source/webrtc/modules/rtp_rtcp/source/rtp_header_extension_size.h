/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_HEADER_EXTENSION_SIZE_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_HEADER_EXTENSION_SIZE_H_

#include "api/array_view.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

namespace webrtc {

struct RtpExtensionSize {
  RTPExtensionType type;
  int value_size;
};

// Calculates rtp header extension size in bytes assuming packet contain
// all `extensions` with provided `value_size`.
// Counts only extensions present among `registered_extensions`.
int RtpHeaderExtensionSize(rtc::ArrayView<const RtpExtensionSize> extensions,
                           const RtpHeaderExtensionMap& registered_extensions);

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_HEADER_EXTENSION_SIZE_H_
