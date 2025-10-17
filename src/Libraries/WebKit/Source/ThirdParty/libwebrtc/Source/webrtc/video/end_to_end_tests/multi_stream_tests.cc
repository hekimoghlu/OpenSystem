/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "call/rtp_config.h"
#include "call/video_receive_stream.h"
#include "call/video_send_stream.h"
#include "rtc_base/event.h"
#include "test/frame_generator_capturer.h"
#include "test/gtest.h"
#include "video/config/video_encoder_config.h"
#include "video/end_to_end_tests/multi_stream_tester.h"

namespace webrtc {
// Each renderer verifies that it receives the expected resolution, and as soon
// as every renderer has received a frame, the test finishes.
TEST(MultiStreamEndToEndTest, SendsAndReceivesMultipleStreams) {
  class VideoOutputObserver : public rtc::VideoSinkInterface<VideoFrame> {
   public:
    VideoOutputObserver(const MultiStreamTester::CodecSettings& settings,
                        uint32_t ssrc,
                        test::FrameGeneratorCapturer** frame_generator)
        : settings_(settings), ssrc_(ssrc), frame_generator_(frame_generator) {}

    void OnFrame(const VideoFrame& video_frame) override {
      EXPECT_EQ(settings_.width, video_frame.width());
      EXPECT_EQ(settings_.height, video_frame.height());
      (*frame_generator_)->Stop();
      done_.Set();
    }

    uint32_t Ssrc() { return ssrc_; }

    bool Wait() { return done_.Wait(TimeDelta::Seconds(30)); }

   private:
    const MultiStreamTester::CodecSettings& settings_;
    const uint32_t ssrc_;
    test::FrameGeneratorCapturer** const frame_generator_;
    rtc::Event done_;
  };

  class Tester : public MultiStreamTester {
   public:
    Tester() = default;
    ~Tester() override = default;

   protected:
    void Wait() override {
      for (const auto& observer : observers_) {
        EXPECT_TRUE(observer->Wait())
            << "Time out waiting for from on ssrc " << observer->Ssrc();
      }
    }

    void UpdateSendConfig(
        size_t stream_index,
        VideoSendStream::Config* send_config,
        VideoEncoderConfig* encoder_config,
        test::FrameGeneratorCapturer** frame_generator) override {
      observers_[stream_index] = std::make_unique<VideoOutputObserver>(
          codec_settings[stream_index], send_config->rtp.ssrcs.front(),
          frame_generator);
    }

    void UpdateReceiveConfig(
        size_t stream_index,
        VideoReceiveStreamInterface::Config* receive_config) override {
      receive_config->renderer = observers_[stream_index].get();
    }

   private:
    std::unique_ptr<VideoOutputObserver> observers_[kNumStreams];
  } tester;

  tester.RunTest();
}
}  // namespace webrtc
