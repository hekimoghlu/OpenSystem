/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
#include <limits>
#include <vector>

#include "api/rtp_parameters.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "api/video/video_source_interface.h"
#include "call/video_receive_stream.h"
#include "call/video_send_stream.h"
#include "rtc_base/checks.h"
#include "rtc_base/event.h"
#include "test/call_test.h"
#include "test/field_trial.h"
#include "test/frame_generator_capturer.h"
#include "test/gtest.h"
#include "test/video_test_constants.h"
#include "video/config/video_encoder_config.h"

namespace webrtc {
namespace {
constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 28;
}  // namespace

// Minimal normal usage at start, then 60s overuse.
class CpuOveruseTest : public test::CallTest {
 protected:
  CpuOveruseTest()
      : field_trials_("WebRTC-ForceSimulatedOveruseIntervalMs/1-60000-60000/") {
  }

  void RunTestAndCheckForAdaptation(
      const DegradationPreference& degradation_preference,
      bool expect_adaptation);

  test::ScopedFieldTrials field_trials_;
};

void CpuOveruseTest::RunTestAndCheckForAdaptation(
    const DegradationPreference& degradation_preference,
    bool expect_adaptation) {
  class OveruseObserver
      : public test::SendTest,
        public test::FrameGeneratorCapturer::SinkWantsObserver {
   public:
    OveruseObserver(const DegradationPreference& degradation_preference,
                    bool expect_adaptation)
        : SendTest(expect_adaptation
                       ? test::VideoTestConstants::kLongTimeout
                       : test::VideoTestConstants::kDefaultTimeout),
          degradation_preference_(degradation_preference),
          expect_adaptation_(expect_adaptation) {}

   private:
    void OnFrameGeneratorCapturerCreated(
        test::FrameGeneratorCapturer* frame_generator_capturer) override {
      frame_generator_capturer->SetSinkWantsObserver(this);
      // Set initial resolution.
      frame_generator_capturer->ChangeResolution(kWidth, kHeight);
    }

    // Called when FrameGeneratorCapturer::AddOrUpdateSink is called.
    void OnSinkWantsChanged(rtc::VideoSinkInterface<VideoFrame>* sink,
                            const rtc::VideoSinkWants& wants) override {
      if (wants.max_pixel_count == std::numeric_limits<int>::max() &&
          wants.max_framerate_fps == kFps) {
        // Max configured framerate is initially set.
        return;
      }
      switch (degradation_preference_) {
        case DegradationPreference::MAINTAIN_FRAMERATE:
          EXPECT_LT(wants.max_pixel_count, kWidth * kHeight);
          observation_complete_.Set();
          break;
        case DegradationPreference::MAINTAIN_RESOLUTION:
          EXPECT_LT(wants.max_framerate_fps, kFps);
          observation_complete_.Set();
          break;
        case DegradationPreference::BALANCED:
          if (wants.max_pixel_count == std::numeric_limits<int>::max() &&
              wants.max_framerate_fps == std::numeric_limits<int>::max()) {
            // `adapt_counters_` map in VideoStreamEncoder is reset when
            // balanced mode is set.
            break;
          }
          EXPECT_TRUE(wants.max_pixel_count < kWidth * kHeight ||
                      wants.max_framerate_fps < kFps);
          observation_complete_.Set();
          break;
        default:
          RTC_DCHECK_NOTREACHED();
      }
    }

    void ModifyVideoConfigs(
        VideoSendStream::Config* send_config,
        std::vector<VideoReceiveStreamInterface::Config>* receive_configs,
        VideoEncoderConfig* encoder_config) override {
      EXPECT_FALSE(encoder_config->simulcast_layers.empty());
      encoder_config->simulcast_layers[0].max_framerate = kFps;
    }

    void ModifyVideoDegradationPreference(
        DegradationPreference* degradation_preference) override {
      *degradation_preference = degradation_preference_;
    }

    void PerformTest() override {
      EXPECT_EQ(expect_adaptation_, Wait())
          << "Timed out while waiting for a scale down.";
    }

    const DegradationPreference degradation_preference_;
    const bool expect_adaptation_;
  } test(degradation_preference, expect_adaptation);

  RunBaseTest(&test);
}

TEST_F(CpuOveruseTest, AdaptsDownInResolutionOnOveruse) {
  RunTestAndCheckForAdaptation(DegradationPreference::MAINTAIN_FRAMERATE, true);
}

TEST_F(CpuOveruseTest, AdaptsDownInFpsOnOveruse) {
  RunTestAndCheckForAdaptation(DegradationPreference::MAINTAIN_RESOLUTION,
                               true);
}

TEST_F(CpuOveruseTest, AdaptsDownInResolutionOrFpsOnOveruse) {
  RunTestAndCheckForAdaptation(DegradationPreference::BALANCED, true);
}

TEST_F(CpuOveruseTest, NoAdaptDownOnOveruse) {
  RunTestAndCheckForAdaptation(DegradationPreference::DISABLED, false);
}
}  // namespace webrtc
