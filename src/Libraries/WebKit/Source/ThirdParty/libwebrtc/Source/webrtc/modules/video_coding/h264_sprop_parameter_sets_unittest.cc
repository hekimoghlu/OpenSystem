/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "modules/video_coding/h264_sprop_parameter_sets.h"

#include <vector>

#include "test/gtest.h"

namespace webrtc {

class H264SpropParameterSetsTest : public ::testing::Test {
 public:
  H264SpropParameterSets h264_sprop;
};

TEST_F(H264SpropParameterSetsTest, Base64DecodeSprop) {
  // Example sprop string from https://tools.ietf.org/html/rfc3984 .
  EXPECT_TRUE(h264_sprop.DecodeSprop("Z0IACpZTBYmI,aMljiA=="));
  static const std::vector<uint8_t> raw_sps{0x67, 0x42, 0x00, 0x0A, 0x96,
                                            0x53, 0x05, 0x89, 0x88};
  static const std::vector<uint8_t> raw_pps{0x68, 0xC9, 0x63, 0x88};
  EXPECT_EQ(raw_sps, h264_sprop.sps_nalu());
  EXPECT_EQ(raw_pps, h264_sprop.pps_nalu());
}

TEST_F(H264SpropParameterSetsTest, InvalidData) {
  EXPECT_FALSE(h264_sprop.DecodeSprop(","));
  EXPECT_FALSE(h264_sprop.DecodeSprop(""));
  EXPECT_FALSE(h264_sprop.DecodeSprop(",iA=="));
  EXPECT_FALSE(h264_sprop.DecodeSprop("iA==,"));
  EXPECT_TRUE(h264_sprop.DecodeSprop("iA==,iA=="));
  EXPECT_FALSE(h264_sprop.DecodeSprop("--,--"));
  EXPECT_FALSE(h264_sprop.DecodeSprop(",,"));
  EXPECT_FALSE(h264_sprop.DecodeSprop("iA=="));
}

}  // namespace webrtc
