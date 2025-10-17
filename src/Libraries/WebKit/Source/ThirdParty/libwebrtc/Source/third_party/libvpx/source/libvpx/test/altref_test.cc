/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include "vpx_config.h"
namespace {

#if CONFIG_VP8_ENCODER

// lookahead range: [kLookAheadMin, kLookAheadMax).
const int kLookAheadMin = 5;
const int kLookAheadMax = 26;

class AltRefTest : public ::libvpx_test::EncoderTest,
                   public ::libvpx_test::CodecTestWithParam<int> {
 protected:
  AltRefTest() : EncoderTest(GET_PARAM(0)), altref_count_(0) {}
  ~AltRefTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(libvpx_test::kTwoPassGood);
  }

  void BeginPassHook(unsigned int /*pass*/) override { altref_count_ = 0; }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(VP8E_SET_CPUUSED, 3);
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->data.frame.flags & VPX_FRAME_IS_INVISIBLE) ++altref_count_;
  }

  int altref_count() const { return altref_count_; }

 private:
  int altref_count_;
};

TEST_P(AltRefTest, MonotonicTimestamps) {
  const vpx_rational timebase = { 33333333, 1000000000 };
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = 1000;
  cfg_.g_lag_in_frames = GET_PARAM(1);

  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                     timebase.den, timebase.num, 0, 30);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  EXPECT_GE(altref_count(), 1);
}

VP8_INSTANTIATE_TEST_SUITE(AltRefTest,
                           ::testing::Range(kLookAheadMin, kLookAheadMax));

#endif  // CONFIG_VP8_ENCODER

class AltRefForcedKeyTestLarge
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, int> {
 protected:
  AltRefForcedKeyTestLarge()
      : EncoderTest(GET_PARAM(0)), encoding_mode_(GET_PARAM(1)),
        cpu_used_(GET_PARAM(2)), forced_kf_frame_num_(1), frame_num_(0) {}
  ~AltRefForcedKeyTestLarge() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);
    cfg_.rc_end_usage = VPX_VBR;
    cfg_.g_threads = 0;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, cpu_used_);
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
#if CONFIG_VP9_ENCODER
      // override test default for tile columns if necessary.
      if (GET_PARAM(0) == &libvpx_test::kVP9) {
        encoder->Control(VP9E_SET_TILE_COLUMNS, 6);
      }
#endif
    }
    frame_flags_ =
        (video->frame() == forced_kf_frame_num_) ? VPX_EFLAG_FORCE_KF : 0;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (frame_num_ == forced_kf_frame_num_) {
      ASSERT_TRUE(!!(pkt->data.frame.flags & VPX_FRAME_IS_KEY))
          << "Frame #" << frame_num_ << " isn't a keyframe!";
    }
    ++frame_num_;
  }

  ::libvpx_test::TestMode encoding_mode_;
  int cpu_used_;
  unsigned int forced_kf_frame_num_;
  unsigned int frame_num_;
};

TEST_P(AltRefForcedKeyTestLarge, Frame1IsKey) {
  const vpx_rational timebase = { 1, 30 };
  const int lag_values[] = { 3, 15, 25, -1 };

  forced_kf_frame_num_ = 1;
  for (int i = 0; lag_values[i] != -1; ++i) {
    frame_num_ = 0;
    cfg_.g_lag_in_frames = lag_values[i];
    libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       timebase.den, timebase.num, 0, 30);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }
}

TEST_P(AltRefForcedKeyTestLarge, ForcedFrameIsKey) {
  const vpx_rational timebase = { 1, 30 };
  const int lag_values[] = { 3, 15, 25, -1 };

  for (int i = 0; lag_values[i] != -1; ++i) {
    frame_num_ = 0;
    forced_kf_frame_num_ = lag_values[i] - 1;
    cfg_.g_lag_in_frames = lag_values[i];
    libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       timebase.den, timebase.num, 0, 30);
    ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  }
}

VP8_INSTANTIATE_TEST_SUITE(AltRefForcedKeyTestLarge,
                           ::testing::Values(::libvpx_test::kOnePassGood),
                           ::testing::Range(0, 9));

VP9_INSTANTIATE_TEST_SUITE(AltRefForcedKeyTestLarge,
                           ::testing::Values(::libvpx_test::kOnePassGood),
                           ::testing::Range(0, 9));
}  // namespace
