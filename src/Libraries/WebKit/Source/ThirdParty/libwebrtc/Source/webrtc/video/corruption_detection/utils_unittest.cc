/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "video/corruption_detection/utils.h"

#include "api/video/video_codec_type.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

#if GTEST_HAS_DEATH_TEST
using ::testing::_;
#endif  // GTEST_HAS_DEATH_TEST

TEST(UtilsTest, FindCodecFromString) {
  EXPECT_EQ(GetVideoCodecType(/*codec_name=*/"VP8"), kVideoCodecVP8);
  EXPECT_EQ(GetVideoCodecType(/*codec_name=*/"libvpx-vp9"), kVideoCodecVP9);
  EXPECT_EQ(GetVideoCodecType(/*codec_name=*/"ImprovedAV1"), kVideoCodecAV1);
  EXPECT_EQ(GetVideoCodecType(/*codec_name=*/"lets_use_h264"), kVideoCodecH264);
}

#if GTEST_HAS_DEATH_TEST
TEST(UtilsTest, IfCodecDoesNotExistRaiseError) {
  EXPECT_DEATH(GetVideoCodecType(/*codec_name=*/"Not_a_codec"), _);
}
#endif  // GTEST_HAS_DEATH_TEST

}  // namespace
}  // namespace webrtc
