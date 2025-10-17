/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "test/i420_video_source.h"
#include "test/util.h"

namespace {

class AltRefAqSegmentTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, int> {
 protected:
  AltRefAqSegmentTest() : EncoderTest(GET_PARAM(0)) {}
  ~AltRefAqSegmentTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
    set_cpu_used_ = GET_PARAM(2);
    aq_mode_ = 0;
    alt_ref_aq_mode_ = 0;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
      encoder->Control(VP9E_SET_ALT_REF_AQ, alt_ref_aq_mode_);
      encoder->Control(VP9E_SET_AQ_MODE, aq_mode_);
      encoder->Control(VP8E_SET_MAX_INTRA_BITRATE_PCT, 100);
    }
  }

  int set_cpu_used_;
  int aq_mode_;
  int alt_ref_aq_mode_;
};

// Validate that this ALT_REF_AQ/AQ segmentation mode
// (ALT_REF_AQ=0, AQ=0/no_aq)
// encodes and decodes without a mismatch.
TEST_P(AltRefAqSegmentTest, TestNoMisMatchAltRefAQ0) {
  cfg_.rc_min_quantizer = 8;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_target_bitrate = 300;

  aq_mode_ = 0;
  alt_ref_aq_mode_ = 1;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// Validate that this ALT_REF_AQ/AQ segmentation mode
// (ALT_REF_AQ=0, AQ=1/variance_aq)
// encodes and decodes without a mismatch.
TEST_P(AltRefAqSegmentTest, TestNoMisMatchAltRefAQ1) {
  cfg_.rc_min_quantizer = 8;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_target_bitrate = 300;

  aq_mode_ = 1;
  alt_ref_aq_mode_ = 1;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// Validate that this ALT_REF_AQ/AQ segmentation mode
// (ALT_REF_AQ=0, AQ=2/complexity_aq)
// encodes and decodes without a mismatch.
TEST_P(AltRefAqSegmentTest, TestNoMisMatchAltRefAQ2) {
  cfg_.rc_min_quantizer = 8;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_target_bitrate = 300;

  aq_mode_ = 2;
  alt_ref_aq_mode_ = 1;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// Validate that this ALT_REF_AQ/AQ segmentation mode
// (ALT_REF_AQ=0, AQ=3/cyclicrefresh_aq)
// encodes and decodes without a mismatch.
TEST_P(AltRefAqSegmentTest, TestNoMisMatchAltRefAQ3) {
  cfg_.rc_min_quantizer = 8;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_target_bitrate = 300;

  aq_mode_ = 3;
  alt_ref_aq_mode_ = 1;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

// Validate that this ALT_REF_AQ/AQ segmentation mode
// (ALT_REF_AQ=0, AQ=4/equator360_aq)
// encodes and decodes without a mismatch.
TEST_P(AltRefAqSegmentTest, TestNoMisMatchAltRefAQ4) {
  cfg_.rc_min_quantizer = 8;
  cfg_.rc_max_quantizer = 56;
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.rc_buf_initial_sz = 500;
  cfg_.rc_buf_optimal_sz = 500;
  cfg_.rc_buf_sz = 1000;
  cfg_.rc_target_bitrate = 300;

  aq_mode_ = 4;
  alt_ref_aq_mode_ = 1;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 100);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

VP9_INSTANTIATE_TEST_SUITE(AltRefAqSegmentTest,
                           ::testing::Values(::libvpx_test::kOnePassGood,
                                             ::libvpx_test::kTwoPassGood),
                           ::testing::Range(2, 5));
}  // namespace
