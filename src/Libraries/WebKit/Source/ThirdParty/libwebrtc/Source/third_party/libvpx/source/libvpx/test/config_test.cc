/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/video_source.h"

namespace {

class ConfigTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  ConfigTest()
      : EncoderTest(GET_PARAM(0)), frame_count_in_(0), frame_count_out_(0),
        frame_count_max_(0) {}
  ~ConfigTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    frame_count_in_ = 0;
    frame_count_out_ = 0;
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource * /*video*/) override {
    ++frame_count_in_;
    abort_ |= (frame_count_in_ >= frame_count_max_);
  }

  void FramePktHook(const vpx_codec_cx_pkt_t * /*pkt*/) override {
    ++frame_count_out_;
  }

  unsigned int frame_count_in_;
  unsigned int frame_count_out_;
  unsigned int frame_count_max_;
};

TEST_P(ConfigTest, LagIsDisabled) {
  frame_count_max_ = 2;
  cfg_.g_lag_in_frames = 15;

  libvpx_test::DummyVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  EXPECT_EQ(frame_count_in_, frame_count_out_);
}

VP8_INSTANTIATE_TEST_SUITE(ConfigTest, ONE_PASS_TEST_MODES);
}  // namespace
