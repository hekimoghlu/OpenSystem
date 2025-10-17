/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#ifndef TEST_SCENARIO_AUDIO_STREAM_H_
#define TEST_SCENARIO_AUDIO_STREAM_H_
#include <memory>
#include <string>
#include <vector>

#include "test/scenario/call_client.h"
#include "test/scenario/column_printer.h"
#include "test/scenario/network_node.h"
#include "test/scenario/scenario_config.h"

namespace webrtc {
namespace test {

// SendAudioStream represents sending of audio. It can be used for starting the
// stream if neccessary.
class SendAudioStream {
 public:
  ~SendAudioStream();

  SendAudioStream(const SendAudioStream&) = delete;
  SendAudioStream& operator=(const SendAudioStream&) = delete;

  void Start();
  void Stop();
  void SetMuted(bool mute);
  ColumnPrinter StatsPrinter();

 private:
  friend class Scenario;
  friend class AudioStreamPair;
  friend class ReceiveAudioStream;
  SendAudioStream(CallClient* sender,
                  AudioStreamConfig config,
                  rtc::scoped_refptr<AudioEncoderFactory> encoder_factory,
                  Transport* send_transport);
  AudioSendStream* send_stream_ = nullptr;
  CallClient* const sender_;
  const AudioStreamConfig config_;
  uint32_t ssrc_;
};

// ReceiveAudioStream represents an audio receiver. It can't be used directly.
class ReceiveAudioStream {
 public:
  ~ReceiveAudioStream();

  ReceiveAudioStream(const ReceiveAudioStream&) = delete;
  ReceiveAudioStream& operator=(const ReceiveAudioStream&) = delete;

  void Start();
  void Stop();
  AudioReceiveStreamInterface::Stats GetStats() const;

 private:
  friend class Scenario;
  friend class AudioStreamPair;
  ReceiveAudioStream(CallClient* receiver,
                     AudioStreamConfig config,
                     SendAudioStream* send_stream,
                     rtc::scoped_refptr<AudioDecoderFactory> decoder_factory,
                     Transport* feedback_transport);
  AudioReceiveStreamInterface* receive_stream_ = nullptr;
  CallClient* const receiver_;
  const AudioStreamConfig config_;
};

// AudioStreamPair represents an audio streaming session. It can be used to
// access underlying send and receive classes. It can also be used in calls to
// the Scenario class.
class AudioStreamPair {
 public:
  ~AudioStreamPair();

  AudioStreamPair(const AudioStreamPair&) = delete;
  AudioStreamPair& operator=(const AudioStreamPair&) = delete;

  SendAudioStream* send() { return &send_stream_; }
  ReceiveAudioStream* receive() { return &receive_stream_; }

 private:
  friend class Scenario;
  AudioStreamPair(CallClient* sender,
                  rtc::scoped_refptr<AudioEncoderFactory> encoder_factory,
                  CallClient* receiver,
                  rtc::scoped_refptr<AudioDecoderFactory> decoder_factory,
                  AudioStreamConfig config);

 private:
  const AudioStreamConfig config_;
  SendAudioStream send_stream_;
  ReceiveAudioStream receive_stream_;
};

std::vector<RtpExtension> GetAudioRtpExtensions(
    const AudioStreamConfig& config);

}  // namespace test
}  // namespace webrtc

#endif  // TEST_SCENARIO_AUDIO_STREAM_H_
