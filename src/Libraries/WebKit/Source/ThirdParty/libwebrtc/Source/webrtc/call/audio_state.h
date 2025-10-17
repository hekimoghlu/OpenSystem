/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
#ifndef CALL_AUDIO_STATE_H_
#define CALL_AUDIO_STATE_H_

#include "api/audio/audio_device.h"
#include "api/audio/audio_mixer.h"
#include "api/audio/audio_processing.h"
#include "api/ref_count.h"
#include "api/scoped_refptr.h"
#include "modules/async_audio_processing/async_audio_processing.h"

namespace webrtc {

class AudioTransport;

// AudioState holds the state which must be shared between multiple instances of
// webrtc::Call for audio processing purposes.
class AudioState : public RefCountInterface {
 public:
  struct Config {
    Config();
    ~Config();

    // The audio mixer connected to active receive streams. One per
    // AudioState.
    rtc::scoped_refptr<AudioMixer> audio_mixer;

    // The audio processing module.
    rtc::scoped_refptr<webrtc::AudioProcessing> audio_processing;

    // TODO(solenberg): Temporary: audio device module.
    rtc::scoped_refptr<webrtc::AudioDeviceModule> audio_device_module;

    rtc::scoped_refptr<AsyncAudioProcessing::Factory>
        async_audio_processing_factory;
  };

  virtual AudioProcessing* audio_processing() = 0;
  virtual AudioTransport* audio_transport() = 0;

  // Enable/disable playout of the audio channels. Enabled by default.
  // This will stop playout of the underlying audio device but start a task
  // which will poll for audio data every 10ms to ensure that audio processing
  // happens and the audio stats are updated.
  virtual void SetPlayout(bool enabled) = 0;

  // Enable/disable recording of the audio channels. Enabled by default.
  // This will stop recording of the underlying audio device and no audio
  // packets will be encoded or transmitted.
  virtual void SetRecording(bool enabled) = 0;

  virtual void SetStereoChannelSwapping(bool enable) = 0;

  static rtc::scoped_refptr<AudioState> Create(
      const AudioState::Config& config);

  ~AudioState() override {}
};
}  // namespace webrtc

#endif  // CALL_AUDIO_STATE_H_
