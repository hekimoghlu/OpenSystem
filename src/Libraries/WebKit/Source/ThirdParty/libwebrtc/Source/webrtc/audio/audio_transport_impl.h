/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#ifndef AUDIO_AUDIO_TRANSPORT_IMPL_H_
#define AUDIO_AUDIO_TRANSPORT_IMPL_H_

#include <memory>
#include <vector>

#include "api/audio/audio_device.h"
#include "api/audio/audio_mixer.h"
#include "api/audio/audio_processing.h"
#include "api/scoped_refptr.h"
#include "common_audio/resampler/include/push_resampler.h"
#include "modules/async_audio_processing/async_audio_processing.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class AudioSender;

class AudioTransportImpl : public AudioTransport {
 public:
  AudioTransportImpl(
      AudioMixer* mixer,
      AudioProcessing* audio_processing,
      AsyncAudioProcessing::Factory* async_audio_processing_factory);

  AudioTransportImpl() = delete;
  AudioTransportImpl(const AudioTransportImpl&) = delete;
  AudioTransportImpl& operator=(const AudioTransportImpl&) = delete;

  ~AudioTransportImpl() override;

  // TODO(bugs.webrtc.org/13620) Deprecate this function
  int32_t RecordedDataIsAvailable(const void* audioSamples,
                                  size_t nSamples,
                                  size_t nBytesPerSample,
                                  size_t nChannels,
                                  uint32_t samplesPerSec,
                                  uint32_t totalDelayMS,
                                  int32_t clockDrift,
                                  uint32_t currentMicLevel,
                                  bool keyPressed,
                                  uint32_t& newMicLevel) override;

  int32_t RecordedDataIsAvailable(
      const void* audioSamples,
      size_t nSamples,
      size_t nBytesPerSample,
      size_t nChannels,
      uint32_t samplesPerSec,
      uint32_t totalDelayMS,
      int32_t clockDrift,
      uint32_t currentMicLevel,
      bool keyPressed,
      uint32_t& newMicLevel,
      std::optional<int64_t> estimated_capture_time_ns) override;

  int32_t NeedMorePlayData(size_t nSamples,
                           size_t nBytesPerSample,
                           size_t nChannels,
                           uint32_t samplesPerSec,
                           void* audioSamples,
                           size_t& nSamplesOut,
                           int64_t* elapsed_time_ms,
                           int64_t* ntp_time_ms) override;

  void PullRenderData(int bits_per_sample,
                      int sample_rate,
                      size_t number_of_channels,
                      size_t number_of_frames,
                      void* audio_data,
                      int64_t* elapsed_time_ms,
                      int64_t* ntp_time_ms) override;

  void UpdateAudioSenders(std::vector<AudioSender*> senders,
                          int send_sample_rate_hz,
                          size_t send_num_channels);
  void SetStereoChannelSwapping(bool enable);

 private:
  void SendProcessedData(std::unique_ptr<AudioFrame> audio_frame);

  // Shared.
  AudioProcessing* audio_processing_ = nullptr;

  // Capture side.

  // Thread-safe.
  const std::unique_ptr<AsyncAudioProcessing> async_audio_processing_;

  mutable Mutex capture_lock_;
  std::vector<AudioSender*> audio_senders_ RTC_GUARDED_BY(capture_lock_);
  int send_sample_rate_hz_ RTC_GUARDED_BY(capture_lock_) = 8000;
  size_t send_num_channels_ RTC_GUARDED_BY(capture_lock_) = 1;
  bool swap_stereo_channels_ RTC_GUARDED_BY(capture_lock_) = false;
  PushResampler<int16_t> capture_resampler_;

  // Render side.

  rtc::scoped_refptr<AudioMixer> mixer_;
  AudioFrame mixed_frame_;
  // Converts mixed audio to the audio device output rate.
  PushResampler<int16_t> render_resampler_;
};
}  // namespace webrtc

#endif  // AUDIO_AUDIO_TRANSPORT_IMPL_H_
