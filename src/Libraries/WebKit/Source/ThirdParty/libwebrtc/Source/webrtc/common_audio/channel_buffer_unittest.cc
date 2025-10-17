/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#include "common_audio/channel_buffer.h"

#include "test/gtest.h"
#include "test/testsupport/rtc_expect_death.h"

namespace webrtc {

namespace {

const size_t kNumFrames = 480u;
const size_t kStereo = 2u;
const size_t kMono = 1u;

void ExpectNumChannels(const IFChannelBuffer& ifchb, size_t num_channels) {
  EXPECT_EQ(ifchb.ibuf_const()->num_channels(), num_channels);
  EXPECT_EQ(ifchb.fbuf_const()->num_channels(), num_channels);
  EXPECT_EQ(ifchb.num_channels(), num_channels);
}

}  // namespace

TEST(ChannelBufferTest, SetNumChannelsSetsNumChannels) {
  ChannelBuffer<float> chb(kNumFrames, kStereo);
  EXPECT_EQ(chb.num_channels(), kStereo);
  chb.set_num_channels(kMono);
  EXPECT_EQ(chb.num_channels(), kMono);
}

TEST(IFChannelBufferTest, SetNumChannelsSetsChannelBuffersNumChannels) {
  IFChannelBuffer ifchb(kNumFrames, kStereo);
  ExpectNumChannels(ifchb, kStereo);
  ifchb.set_num_channels(kMono);
  ExpectNumChannels(ifchb, kMono);
}

TEST(IFChannelBufferTest, SettingNumChannelsOfOneChannelBufferSetsTheOther) {
  IFChannelBuffer ifchb(kNumFrames, kStereo);
  ExpectNumChannels(ifchb, kStereo);
  ifchb.ibuf()->set_num_channels(kMono);
  ExpectNumChannels(ifchb, kMono);
  ifchb.fbuf()->set_num_channels(kStereo);
  ExpectNumChannels(ifchb, kStereo);
}

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)
TEST(ChannelBufferDeathTest, SetNumChannelsDeathTest) {
  ChannelBuffer<float> chb(kNumFrames, kMono);
  RTC_EXPECT_DEATH(chb.set_num_channels(kStereo), "num_channels");
}

TEST(IFChannelBufferDeathTest, SetNumChannelsDeathTest) {
  IFChannelBuffer ifchb(kNumFrames, kMono);
  RTC_EXPECT_DEATH(ifchb.ibuf()->set_num_channels(kStereo), "num_channels");
}
#endif

}  // namespace webrtc
