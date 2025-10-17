/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifndef AUDIO_AUDIO_STATE_H_
#define AUDIO_AUDIO_STATE_H_

#include <map>
#include <memory>

#include "api/sequence_checker.h"
#include "audio/audio_transport_impl.h"
#include "call/audio_state.h"
#include "rtc_base/containers/flat_set.h"
#include "rtc_base/ref_count.h"
#include "rtc_base/task_utils/repeating_task.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class AudioSendStream;
class AudioReceiveStreamInterface;

namespace internal {

class AudioState : public webrtc::AudioState {
 public:
  explicit AudioState(const AudioState::Config& config);

  AudioState() = delete;
  AudioState(const AudioState&) = delete;
  AudioState& operator=(const AudioState&) = delete;

  ~AudioState() override;

  AudioProcessing* audio_processing() override;
  AudioTransport* audio_transport() override;

  void SetPlayout(bool enabled) override;
  void SetRecording(bool enabled) override;

  void SetStereoChannelSwapping(bool enable) override;

  AudioDeviceModule* audio_device_module() {
    RTC_DCHECK(config_.audio_device_module);
    return config_.audio_device_module.get();
  }

  void AddReceivingStream(webrtc::AudioReceiveStreamInterface* stream);
  void RemoveReceivingStream(webrtc::AudioReceiveStreamInterface* stream);

  void AddSendingStream(webrtc::AudioSendStream* stream,
                        int sample_rate_hz,
                        size_t num_channels);
  void RemoveSendingStream(webrtc::AudioSendStream* stream);

 private:
  void UpdateAudioTransportWithSendingStreams();
  void UpdateNullAudioPollerState() RTC_RUN_ON(&thread_checker_);

  SequenceChecker thread_checker_;
  SequenceChecker process_thread_checker_{SequenceChecker::kDetached};
  const webrtc::AudioState::Config config_;
  bool recording_enabled_ = true;
  bool playout_enabled_ = true;

  // Transports mixed audio from the mixer to the audio device and
  // recorded audio to the sending streams.
  AudioTransportImpl audio_transport_;

  // Null audio poller is used to continue polling the audio streams if audio
  // playout is disabled so that audio processing still happens and the audio
  // stats are still updated.
  RepeatingTaskHandle null_audio_poller_ RTC_GUARDED_BY(&thread_checker_);

  webrtc::flat_set<webrtc::AudioReceiveStreamInterface*> receiving_streams_;
  struct StreamProperties {
    int sample_rate_hz = 0;
    size_t num_channels = 0;
  };
  std::map<webrtc::AudioSendStream*, StreamProperties> sending_streams_;
};
}  // namespace internal
}  // namespace webrtc

#endif  // AUDIO_AUDIO_STATE_H_
