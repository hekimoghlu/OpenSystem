/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "modules/rtp_rtcp/source/video_rtp_depacketizer_raw.h"

#include <cstdint>
#include <optional>

#include "rtc_base/copy_on_write_buffer.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

TEST(VideoRtpDepacketizerRaw, PassRtpPayloadAsVideoPayload) {
  const uint8_t kPayload[] = {0x05, 0x25, 0x52};
  rtc::CopyOnWriteBuffer rtp_payload(kPayload);

  VideoRtpDepacketizerRaw depacketizer;
  std::optional<VideoRtpDepacketizer::ParsedRtpPayload> parsed =
      depacketizer.Parse(rtp_payload);

  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->video_payload.size(), rtp_payload.size());
  // Check there was no memcpy involved by verifying return and original buffers
  // point to the same buffer.
  EXPECT_EQ(parsed->video_payload.cdata(), rtp_payload.cdata());
}

TEST(VideoRtpDepacketizerRaw, UsesDefaultValuesForVideoHeader) {
  const uint8_t kPayload[] = {0x05, 0x25, 0x52};
  rtc::CopyOnWriteBuffer rtp_payload(kPayload);

  VideoRtpDepacketizerRaw depacketizer;
  std::optional<VideoRtpDepacketizer::ParsedRtpPayload> parsed =
      depacketizer.Parse(rtp_payload);

  ASSERT_TRUE(parsed);
  EXPECT_FALSE(parsed->video_header.generic);
  EXPECT_EQ(parsed->video_header.codec, kVideoCodecGeneric);
}

}  // namespace
}  // namespace webrtc
