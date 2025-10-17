/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
#include "audio/test/audio_end_to_end_test.h"

#include <algorithm>
#include <memory>

#include "api/task_queue/task_queue_base.h"
#include "call/fake_network_pipe.h"
#include "modules/audio_device/include/test_audio_device.h"
#include "system_wrappers/include/sleep.h"
#include "test/gtest.h"
#include "test/video_test_constants.h"

namespace webrtc {
namespace test {
namespace {

constexpr int kSampleRate = 48000;

}  // namespace

AudioEndToEndTest::AudioEndToEndTest()
    : EndToEndTest(VideoTestConstants::kDefaultTimeout) {}

size_t AudioEndToEndTest::GetNumVideoStreams() const {
  return 0;
}

size_t AudioEndToEndTest::GetNumAudioStreams() const {
  return 1;
}

size_t AudioEndToEndTest::GetNumFlexfecStreams() const {
  return 0;
}

std::unique_ptr<TestAudioDeviceModule::Capturer>
AudioEndToEndTest::CreateCapturer() {
  return TestAudioDeviceModule::CreatePulsedNoiseCapturer(32000, kSampleRate);
}

std::unique_ptr<TestAudioDeviceModule::Renderer>
AudioEndToEndTest::CreateRenderer() {
  return TestAudioDeviceModule::CreateDiscardRenderer(kSampleRate);
}

void AudioEndToEndTest::OnFakeAudioDevicesCreated(
    AudioDeviceModule* send_audio_device,
    AudioDeviceModule* /* recv_audio_device */) {
  send_audio_device_ = send_audio_device;
}

void AudioEndToEndTest::ModifyAudioConfigs(
    AudioSendStream::Config* send_config,
    std::vector<AudioReceiveStreamInterface::Config>* /* receive_configs */) {
  // Large bitrate by default.
  const webrtc::SdpAudioFormat kDefaultFormat("opus", 48000, 2,
                                              {{"stereo", "1"}});
  send_config->send_codec_spec = AudioSendStream::Config::SendCodecSpec(
      test::VideoTestConstants::kAudioSendPayloadType, kDefaultFormat);
  send_config->min_bitrate_bps = 32000;
  send_config->max_bitrate_bps = 32000;
}

void AudioEndToEndTest::OnAudioStreamsCreated(
    AudioSendStream* send_stream,
    const std::vector<AudioReceiveStreamInterface*>& receive_streams) {
  ASSERT_NE(nullptr, send_stream);
  ASSERT_EQ(1u, receive_streams.size());
  ASSERT_NE(nullptr, receive_streams[0]);
  send_stream_ = send_stream;
  receive_stream_ = receive_streams[0];
}

}  // namespace test
}  // namespace webrtc
