/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#include <cstdio>
#include <cstdlib>
#include <string>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/md5_helper.h"
#include "test/util.h"
#include "vpx_mem/vpx_mem.h"

namespace {
class TileIndependenceTest : public ::libvpx_test::EncoderTest,
                             public ::libvpx_test::CodecTestWithParam<int> {
 protected:
  TileIndependenceTest()
      : EncoderTest(GET_PARAM(0)), md5_fw_order_(), md5_inv_order_(),
        n_tiles_(GET_PARAM(1)) {
    init_flags_ = VPX_CODEC_USE_PSNR;
    vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
    cfg.w = 704;
    cfg.h = 144;
    cfg.threads = 1;
    fw_dec_ = codec_->CreateDecoder(cfg, 0);
    inv_dec_ = codec_->CreateDecoder(cfg, 0);
    inv_dec_->Control(VP9_INVERT_TILE_DECODE_ORDER, 1);
  }

  ~TileIndependenceTest() override {
    delete fw_dec_;
    delete inv_dec_;
  }

  void SetUp() override {
    InitializeConfig();
    SetMode(libvpx_test::kTwoPassGood);
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP9E_SET_TILE_COLUMNS, n_tiles_);
    }
  }

  void UpdateMD5(::libvpx_test::Decoder *dec, const vpx_codec_cx_pkt_t *pkt,
                 ::libvpx_test::MD5 *md5) {
    const vpx_codec_err_t res = dec->DecodeFrame(
        reinterpret_cast<uint8_t *>(pkt->data.frame.buf), pkt->data.frame.sz);
    if (res != VPX_CODEC_OK) {
      abort_ = true;
      ASSERT_EQ(VPX_CODEC_OK, res);
    }
    const vpx_image_t *img = dec->GetDxData().Next();
    md5->Add(img);
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    UpdateMD5(fw_dec_, pkt, &md5_fw_order_);
    UpdateMD5(inv_dec_, pkt, &md5_inv_order_);
  }

  ::libvpx_test::MD5 md5_fw_order_, md5_inv_order_;
  ::libvpx_test::Decoder *fw_dec_, *inv_dec_;

 private:
  int n_tiles_;
};

// run an encode with 2 or 4 tiles, and do the decode both in normal and
// inverted tile ordering. Ensure that the MD5 of the output in both cases
// is identical. If so, tiles are considered independent and the test passes.
TEST_P(TileIndependenceTest, MD5Match) {
  const vpx_rational timebase = { 33333333, 1000000000 };
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = 500;
  cfg_.g_lag_in_frames = 25;
  cfg_.rc_end_usage = VPX_VBR;

  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 704, 144,
                                     timebase.den, timebase.num, 0, 30);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  const char *md5_fw_str = md5_fw_order_.Get();
  const char *md5_inv_str = md5_inv_order_.Get();

  // could use ASSERT_EQ(!memcmp(.., .., 16) here, but this gives nicer
  // output if it fails. Not sure if it's helpful since it's really just
  // a MD5...
  ASSERT_STREQ(md5_fw_str, md5_inv_str);
}

VP9_INSTANTIATE_TEST_SUITE(TileIndependenceTest, ::testing::Range(0, 2, 1));
}  // namespace
