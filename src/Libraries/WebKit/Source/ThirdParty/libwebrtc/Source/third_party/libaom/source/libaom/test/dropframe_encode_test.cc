/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"

namespace {

// Params: test mode, threads.
class DropFrameEncodeTestLarge
    : public ::libaom_test::CodecTestWith2Params<libaom_test::TestMode,
                                                 unsigned int>,
      public ::libaom_test::EncoderTest {
 protected:
  DropFrameEncodeTestLarge()
      : EncoderTest(GET_PARAM(0)), frame_number_(0), threads_(GET_PARAM(2)) {}

  void SetUp() override { InitializeConfig(GET_PARAM(1)); }

  void PreEncodeFrameHook(::libaom_test::VideoSource *video,
                          ::libaom_test::Encoder *encoder) override {
    frame_number_ = video->frame();
    if (frame_number_ == 0) {
      encoder->Control(AOME_SET_CPUUSED, 1);
    }
  }

  unsigned int frame_number_;
  unsigned int threads_;
};

// Test to reproduce the assertion failure related to buf->display_idx in
// init_gop_frames_for_tpl() and segmentation fault reported in aomedia:3372
// while encoding with --drop-frame=1.
TEST_P(DropFrameEncodeTestLarge, TestNoMisMatch) {
  cfg_.rc_end_usage = AOM_CBR;
  cfg_.rc_buf_sz = 1;
  cfg_.g_pass = AOM_RC_ONE_PASS;
  cfg_.rc_dropframe_thresh = 1;
  cfg_.g_threads = threads_;

  ::libaom_test::I420VideoSource video("desktopqvga2.320_240.yuv", 320, 240, 30,
                                       1, 0, 100);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

AV1_INSTANTIATE_TEST_SUITE(DropFrameEncodeTestLarge,
                           ::testing::Values(::libaom_test::kOnePassGood),
                           ::testing::Values(1, 4));

}  // namespace
