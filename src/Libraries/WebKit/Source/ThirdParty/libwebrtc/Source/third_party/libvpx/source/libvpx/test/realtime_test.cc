/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#include <limits.h>

#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/video_source.h"
#include "vpx_config.h"

namespace {

const int kVideoSourceWidth = 320;
const int kVideoSourceHeight = 240;
const int kFramesToEncode = 2;

class RealtimeTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  RealtimeTest() : EncoderTest(GET_PARAM(0)), frame_packets_(0) {}
  ~RealtimeTest() override = default;

  void SetUp() override {
    InitializeConfig();
    cfg_.g_lag_in_frames = 0;
    SetMode(::libvpx_test::kRealTime);
  }

  void BeginPassHook(unsigned int /*pass*/) override {
#if !CONFIG_REALTIME_ONLY
    // TODO(tomfinegan): We're changing the pass value here to make sure
    // we get frames when real time mode is combined with |g_pass| set to
    // VPX_RC_FIRST_PASS. This is necessary because EncoderTest::RunLoop() sets
    // the pass value based on the mode passed into EncoderTest::SetMode(),
    // which overrides the one specified in SetUp() above.
    cfg_.g_pass = VPX_RC_FIRST_PASS;
#endif
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0 && set_cpu_used_) {
      encoder->Control(VP8E_SET_CPUUSED, 8);
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t * /*pkt*/) override {
    frame_packets_++;
  }

  bool IsVP9() const {
#if CONFIG_VP9_ENCODER
    return codec_ == &libvpx_test::kVP9;
#else
    return false;
#endif
  }

  void TestIntegerOverflow(unsigned int width, unsigned int height) {
    ::libvpx_test::RandomVideoSource video;
    video.SetSize(width, height);
    video.set_limit(20);
    cfg_.rc_target_bitrate = UINT_MAX;
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }

  void TestEncode() {
    ::libvpx_test::RandomVideoSource video;
    video.SetSize(kVideoSourceWidth, kVideoSourceHeight);
    video.set_limit(kFramesToEncode);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
    EXPECT_EQ(kFramesToEncode, frame_packets_);
  }

  int frame_packets_;
  bool set_cpu_used_ = true;
};

TEST_P(RealtimeTest, RealtimeFirstPassProducesFrames) { TestEncode(); }

TEST_P(RealtimeTest, RealtimeDefaultCpuUsed) {
  set_cpu_used_ = false;
  TestEncode();
}

TEST_P(RealtimeTest, IntegerOverflow) { TestIntegerOverflow(2048, 2048); }

TEST_P(RealtimeTest, IntegerOverflowLarge) {
#ifdef CHROMIUM
  GTEST_SKIP() << "16K framebuffers are not supported by Chromium's allocator.";
#else
  if (IsVP9()) {
#if VPX_ARCH_AARCH64 || VPX_ARCH_X86_64
    TestIntegerOverflow(16384, 16384);
#else
    TestIntegerOverflow(4096, 4096);
#endif
  } else {
    GTEST_SKIP()
        << "TODO(https://crbug.com/webm/1748,https://crbug.com/webm/1751):"
        << " Enable this test after bitstream errors & undefined sanitizer "
           "warnings are fixed.";
    // TestIntegerOverflow(16383, 16383);
  }
#endif  // defined(CHROMIUM)
}

VP8_INSTANTIATE_TEST_SUITE(RealtimeTest,
                           ::testing::Values(::libvpx_test::kRealTime));
VP9_INSTANTIATE_TEST_SUITE(RealtimeTest,
                           ::testing::Values(::libvpx_test::kRealTime));

}  // namespace
