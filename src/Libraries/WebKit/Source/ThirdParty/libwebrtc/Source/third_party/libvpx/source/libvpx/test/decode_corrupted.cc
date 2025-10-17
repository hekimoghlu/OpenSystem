/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include <tuple>

#include "gtest/gtest.h"

#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/i420_video_source.h"
#include "vpx_config.h"
#include "vpx_mem/vpx_mem.h"

namespace {

class DecodeCorruptedFrameTest
    : public ::libvpx_test::EncoderTest,
      public ::testing::TestWithParam<
          std::tuple<const libvpx_test::CodecFactory *> > {
 public:
  DecodeCorruptedFrameTest() : EncoderTest(GET_PARAM(0)) {}

 protected:
  ~DecodeCorruptedFrameTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 600;

    // Set small key frame distance such that we insert more key frames.
    cfg_.kf_max_dist = 3;
    dec_cfg_.threads = 1;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) encoder->Control(VP8E_SET_CPUUSED, 7);
  }

  void MismatchHook(const vpx_image_t * /*img1*/,
                    const vpx_image_t * /*img2*/) override {}

  const vpx_codec_cx_pkt_t *MutateEncoderOutputHook(
      const vpx_codec_cx_pkt_t *pkt) override {
    // Don't edit frame packet on key frame.
    if (pkt->data.frame.flags & VPX_FRAME_IS_KEY) return pkt;
    if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) return pkt;

    memcpy(&modified_pkt_, pkt, sizeof(*pkt));

    // Halve the size so it's corrupted to decoder.
    modified_pkt_.data.frame.sz = modified_pkt_.data.frame.sz / 2;

    return &modified_pkt_;
  }

  bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                          const libvpx_test::VideoSource & /*video*/,
                          libvpx_test::Decoder *decoder) override {
    EXPECT_NE(res_dec, VPX_CODEC_MEM_ERROR) << decoder->DecodeError();
    return VPX_CODEC_MEM_ERROR != res_dec;
  }

  vpx_codec_cx_pkt_t modified_pkt_;
};

TEST_P(DecodeCorruptedFrameTest, DecodeCorruptedFrame) {
  cfg_.rc_target_bitrate = 200;
  cfg_.g_error_resilient = 0;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 300);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

#if CONFIG_VP9
INSTANTIATE_TEST_SUITE_P(
    VP9, DecodeCorruptedFrameTest,
    ::testing::Values(
        static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)));
#endif  // CONFIG_VP9

#if CONFIG_VP8
INSTANTIATE_TEST_SUITE_P(
    VP8, DecodeCorruptedFrameTest,
    ::testing::Values(
        static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP8)));
#endif  // CONFIG_VP8

}  // namespace
