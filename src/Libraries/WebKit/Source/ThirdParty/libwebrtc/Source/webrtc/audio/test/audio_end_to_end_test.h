/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#ifndef AUDIO_TEST_AUDIO_END_TO_END_TEST_H_
#define AUDIO_TEST_AUDIO_END_TO_END_TEST_H_

#include <memory>
#include <string>
#include <vector>

#include "api/audio/audio_device.h"
#include "api/task_queue/task_queue_base.h"
#include "api/test/simulated_network.h"
#include "modules/audio_device/include/test_audio_device.h"
#include "test/call_test.h"

namespace webrtc {
namespace test {

class AudioEndToEndTest : public test::EndToEndTest {
 public:
  AudioEndToEndTest();

 protected:
  AudioDeviceModule* send_audio_device() { return send_audio_device_; }
  const AudioSendStream* send_stream() const { return send_stream_; }
  const AudioReceiveStreamInterface* receive_stream() const {
    return receive_stream_;
  }

  size_t GetNumVideoStreams() const override;
  size_t GetNumAudioStreams() const override;
  size_t GetNumFlexfecStreams() const override;

  std::unique_ptr<TestAudioDeviceModule::Capturer> CreateCapturer() override;
  std::unique_ptr<TestAudioDeviceModule::Renderer> CreateRenderer() override;

  void OnFakeAudioDevicesCreated(AudioDeviceModule* send_audio_device,
                                 AudioDeviceModule* recv_audio_device) override;

  void ModifyAudioConfigs(AudioSendStream::Config* send_config,
                          std::vector<AudioReceiveStreamInterface::Config>*
                              receive_configs) override;
  void OnAudioStreamsCreated(AudioSendStream* send_stream,
                             const std::vector<AudioReceiveStreamInterface*>&
                                 receive_streams) override;

 private:
  AudioDeviceModule* send_audio_device_ = nullptr;
  AudioSendStream* send_stream_ = nullptr;
  AudioReceiveStreamInterface* receive_stream_ = nullptr;
};

}  // namespace test
}  // namespace webrtc

#endif  // AUDIO_TEST_AUDIO_END_TO_END_TEST_H_
