/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#ifndef API_VIDEO_TEST_VIDEO_FRAME_MATCHERS_H_
#define API_VIDEO_TEST_VIDEO_FRAME_MATCHERS_H_

#include "api/rtp_packet_infos.h"
#include "api/video/video_frame.h"
#include "test/gmock.h"

namespace webrtc::test::video_frame_matchers {

MATCHER_P(Rotation, rotation, "") {
  return ::testing::Matches(::testing::Eq(rotation))(arg.rotation());
}

MATCHER_P(NtpTimestamp, ntp_ts, "") {
  return arg.ntp_time_ms() == ntp_ts.ms();
}

MATCHER_P(PacketInfos, m, "") {
  return ::testing::Matches(m)(arg.packet_infos());
}

}  // namespace webrtc::test::video_frame_matchers

#endif  // API_VIDEO_TEST_VIDEO_FRAME_MATCHERS_H_
