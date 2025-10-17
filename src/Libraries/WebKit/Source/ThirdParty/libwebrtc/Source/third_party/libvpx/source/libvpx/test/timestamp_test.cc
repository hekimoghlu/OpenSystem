/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#include "vpx_config.h"

namespace {

const int kVideoSourceWidth = 320;
const int kVideoSourceHeight = 240;
const int kFramesToEncode = 3;

// A video source that exposes functions to set the timebase, framerate and
// starting pts.
class DummyTimebaseVideoSource : public ::libvpx_test::DummyVideoSource {
 public:
  // Parameters num and den set the timebase for the video source.
  DummyTimebaseVideoSource(int num, int den)
      : timebase_({ num, den }), framerate_numerator_(30),
        framerate_denominator_(1), starting_pts_(0) {
    SetSize(kVideoSourceWidth, kVideoSourceHeight);
    set_limit(kFramesToEncode);
  }

  void SetFramerate(int numerator, int denominator) {
    framerate_numerator_ = numerator;
    framerate_denominator_ = denominator;
  }

  // Returns one frames duration in timebase units as a double.
  double FrameDuration() const {
    return (static_cast<double>(timebase_.den) / timebase_.num) /
           (static_cast<double>(framerate_numerator_) / framerate_denominator_);
  }

  vpx_codec_pts_t pts() const override {
    return static_cast<vpx_codec_pts_t>(frame_ * FrameDuration() +
                                        starting_pts_ + 0.5);
  }

  unsigned long duration() const override {
    return static_cast<unsigned long>(FrameDuration() + 0.5);
  }

  vpx_rational_t timebase() const override { return timebase_; }

  void set_starting_pts(int64_t starting_pts) { starting_pts_ = starting_pts; }

 private:
  vpx_rational_t timebase_;
  int framerate_numerator_;
  int framerate_denominator_;
  int64_t starting_pts_;
};

class TimestampTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  TimestampTest() : EncoderTest(GET_PARAM(0)) {}
  ~TimestampTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
  }
};

// Tests encoding in millisecond timebase.
TEST_P(TimestampTest, EncodeFrames) {
  DummyTimebaseVideoSource video(1, 1000);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(TimestampTest, TestMicrosecondTimebase) {
  // Set the timebase to microseconds.
  DummyTimebaseVideoSource video(1, 1000000);
  video.set_limit(1);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(TimestampTest, TestVpxRollover) {
  DummyTimebaseVideoSource video(1, 1000);
  video.set_starting_pts(922337170351ll);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

#if CONFIG_REALTIME_ONLY
VP8_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kRealTime));
VP9_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kRealTime));
#else
VP8_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kTwoPassGood));
VP9_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kTwoPassGood));
#endif
}  // namespace
