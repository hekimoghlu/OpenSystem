/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_ADAPTIVE_DIGITAL_GAIN_CONTROLLER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_ADAPTIVE_DIGITAL_GAIN_CONTROLLER_H_

#include <vector>

#include "api/audio/audio_processing.h"
#include "api/audio/audio_view.h"
#include "modules/audio_processing/agc2/gain_applier.h"

namespace webrtc {

class ApmDataDumper;

// Selects the target digital gain, decides when and how quickly to adapt to the
// target and applies the current gain to 10 ms frames.
class AdaptiveDigitalGainController {
 public:
  // Information about a frame to process.
  struct FrameInfo {
    float speech_probability;    // Probability of speech in the [0, 1] range.
    float speech_level_dbfs;     // Estimated speech level (dBFS).
    bool speech_level_reliable;  // True with reliable speech level estimation.
    float noise_rms_dbfs;        // Estimated noise RMS level (dBFS).
    float headroom_db;           // Headroom (dB).
    // TODO(bugs.webrtc.org/7494): Remove `limiter_envelope_dbfs`.
    float limiter_envelope_dbfs;  // Envelope level from the limiter (dBFS).
  };

  AdaptiveDigitalGainController(
      ApmDataDumper* apm_data_dumper,
      const AudioProcessing::Config::GainController2::AdaptiveDigital& config,
      int adjacent_speech_frames_threshold);
  AdaptiveDigitalGainController(const AdaptiveDigitalGainController&) = delete;
  AdaptiveDigitalGainController& operator=(
      const AdaptiveDigitalGainController&) = delete;

  // Analyzes `info`, updates the digital gain and applies it to a 10 ms
  // `frame`. Supports any sample rate supported by APM.
  void Process(const FrameInfo& info, DeinterleavedView<float> frame);

 private:
  ApmDataDumper* const apm_data_dumper_;
  GainApplier gain_applier_;

  const AudioProcessing::Config::GainController2::AdaptiveDigital config_;
  const int adjacent_speech_frames_threshold_;
  const float max_gain_change_db_per_10ms_;

  int calls_since_last_gain_log_;
  int frames_to_gain_increase_allowed_;
  float last_gain_db_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_ADAPTIVE_DIGITAL_GAIN_CONTROLLER_H_
