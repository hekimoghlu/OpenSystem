/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#ifndef CALL_TEST_MOCK_AUDIO_SEND_STREAM_H_
#define CALL_TEST_MOCK_AUDIO_SEND_STREAM_H_

#include <memory>

#include "call/audio_send_stream.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockAudioSendStream : public AudioSendStream {
 public:
  MOCK_METHOD(const webrtc::AudioSendStream::Config&,
              GetConfig,
              (),
              (const, override));
  MOCK_METHOD(void,
              Reconfigure,
              (const Config& config, SetParametersCallback callback),
              (override));
  MOCK_METHOD(void, Start, (), (override));
  MOCK_METHOD(void, Stop, (), (override));
  // GMock doesn't like move-only types, such as std::unique_ptr.
  void SendAudioData(std::unique_ptr<webrtc::AudioFrame> audio_frame) override {
    SendAudioDataForMock(audio_frame.get());
  }
  MOCK_METHOD(void, SendAudioDataForMock, (webrtc::AudioFrame*));
  MOCK_METHOD(
      bool,
      SendTelephoneEvent,
      (int payload_type, int payload_frequency, int event, int duration_ms),
      (override));
  MOCK_METHOD(void, SetMuted, (bool muted), (override));
  MOCK_METHOD(Stats, GetStats, (), (const, override));
  MOCK_METHOD(Stats, GetStats, (bool has_remote_tracks), (const, override));
};
}  // namespace test
}  // namespace webrtc

#endif  // CALL_TEST_MOCK_AUDIO_SEND_STREAM_H_
