/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_VOICE_ACTIVITY_DETECTOR_H_
#define MODULES_AUDIO_PROCESSING_VAD_VOICE_ACTIVITY_DETECTOR_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "common_audio/resampler/include/resampler.h"
#include "modules/audio_processing/vad/common.h"
#include "modules/audio_processing/vad/pitch_based_vad.h"
#include "modules/audio_processing/vad/standalone_vad.h"
#include "modules/audio_processing/vad/vad_audio_proc.h"

namespace webrtc {

// A Voice Activity Detector (VAD) that combines the voice probability from the
// StandaloneVad and PitchBasedVad to get a more robust estimation.
class VoiceActivityDetector {
 public:
  VoiceActivityDetector();
  ~VoiceActivityDetector();

  // Processes each audio chunk and estimates the voice probability.
  // TODO(bugs.webrtc.org/7494): Switch to rtc::ArrayView and remove
  // `sample_rate_hz`.
  void ProcessChunk(const int16_t* audio, size_t length, int sample_rate_hz);

  // Returns a vector of voice probabilities for each chunk. It can be empty for
  // some chunks, but it catches up afterwards returning multiple values at
  // once.
  const std::vector<double>& chunkwise_voice_probabilities() const {
    return chunkwise_voice_probabilities_;
  }

  // Returns a vector of RMS values for each chunk. It has the same length as
  // chunkwise_voice_probabilities().
  const std::vector<double>& chunkwise_rms() const { return chunkwise_rms_; }

  // Returns the last voice probability, regardless of the internal
  // implementation, although it has a few chunks of delay.
  float last_voice_probability() const { return last_voice_probability_; }

 private:
  // TODO(aluebs): Change these to float.
  std::vector<double> chunkwise_voice_probabilities_;
  std::vector<double> chunkwise_rms_;

  float last_voice_probability_;

  Resampler resampler_;
  VadAudioProc audio_processing_;

  std::unique_ptr<StandaloneVad> standalone_vad_;
  PitchBasedVad pitch_based_vad_;

  int16_t resampled_[kLength10Ms];
  AudioFeatures features_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_VAD_VOICE_ACTIVITY_DETECTOR_H_
