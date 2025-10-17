/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#include "common_video/h265/h265_vps_parser.h"

#include "common_video/h265/h265_common.h"
#include "rtc_base/arraysize.h"
#include "rtc_base/bit_buffer.h"
#include "rtc_base/buffer.h"
#include "test/gtest.h"

namespace webrtc {

// Example VPS can be generated with ffmpeg. Here's an example set of commands,
// runnable on Linux:
// 1) Generate a video, from the camera:
// ffmpeg -i /dev/video0 -r 30 -c:v libx265 -s 1280x720 camera.h265
//
// 2) Open camera.h265 and find the VPS, generally everything between the first
// and second start codes (0 0 0 1 or 0 0 1). The first two bytes should be 0x40
// and 0x01, which should be stripped out before being passed to the parser.

class H265VpsParserTest : public ::testing::Test {
 public:
  H265VpsParserTest() {}
  ~H265VpsParserTest() override {}

  std::optional<H265VpsParser::VpsState> vps_;
};

TEST_F(H265VpsParserTest, TestSampleVPSId) {
  // VPS id 1
  const uint8_t buffer[] = {
      0x1c, 0x01, 0xff, 0xff, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x9d,
      0x08, 0x00, 0x00, 0x03, 0x00, 0x00, 0x78, 0x95, 0x98, 0x09,
  };
  EXPECT_TRUE(static_cast<bool>(vps_ = H265VpsParser::ParseVps(buffer)));
  EXPECT_EQ(1u, vps_->id);
}

}  // namespace webrtc
