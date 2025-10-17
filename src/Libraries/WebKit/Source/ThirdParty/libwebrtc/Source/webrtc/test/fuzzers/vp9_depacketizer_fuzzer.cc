/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_vp9.h"

namespace webrtc {
void FuzzOneInput(const uint8_t* data, size_t size) {
  RTPVideoHeader video_header;
  VideoRtpDepacketizerVp9::ParseRtpPayload(rtc::MakeArrayView(data, size),
                                           &video_header);
}
}  // namespace webrtc
