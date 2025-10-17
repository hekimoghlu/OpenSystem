/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#include <climits>
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "vpx_config.h"

namespace {

class BordersTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  BordersTest() : EncoderTest(GET_PARAM(0)) {}
  ~BordersTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, 1);
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
      encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
      encoder->Control(VP8E_SET_ARNR_TYPE, 3);
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->data.frame.flags & VPX_FRAME_IS_KEY) {
    }
  }
};

TEST_P(BordersTest, TestEncodeHighBitrate) {
  // Validate that this non multiple of 64 wide clip encodes and decodes
  // without a mismatch when passing in a very low max q.  This pushes
  // the encoder to producing lots of big partitions which will likely
  // extend into the border and test the border condition.
  cfg_.g_lag_in_frames = 25;
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 2000;
  cfg_.rc_max_quantizer = 10;

  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       40);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}
TEST_P(BordersTest, TestLowBitrate) {
  // Validate that this clip encodes and decodes without a mismatch
  // when passing in a very high min q.  This pushes the encoder to producing
  // lots of small partitions which might will test the other condition.

  cfg_.g_lag_in_frames = 25;
  cfg_.rc_2pass_vbr_minsection_pct = 5;
  cfg_.rc_2pass_vbr_maxsection_pct = 2000;
  cfg_.rc_target_bitrate = 200;
  cfg_.rc_min_quantizer = 40;

  ::libvpx_test::I420VideoSource video("hantro_odd.yuv", 208, 144, 30, 1, 0,
                                       40);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

#if CONFIG_REALTIME_ONLY
VP9_INSTANTIATE_TEST_SUITE(BordersTest,
                           ::testing::Values(::libvpx_test::kRealTime));
#else
VP9_INSTANTIATE_TEST_SUITE(BordersTest,
                           ::testing::Values(::libvpx_test::kTwoPassGood));
#endif
}  // namespace
