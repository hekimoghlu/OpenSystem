/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_NOISE_SUPPRESSOR_H_
#define MODULES_AUDIO_PROCESSING_NS_NOISE_SUPPRESSOR_H_

#include <memory>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/ns/noise_estimator.h"
#include "modules/audio_processing/ns/ns_common.h"
#include "modules/audio_processing/ns/ns_config.h"
#include "modules/audio_processing/ns/ns_fft.h"
#include "modules/audio_processing/ns/speech_probability_estimator.h"
#include "modules/audio_processing/ns/wiener_filter.h"

namespace webrtc {

// Class for suppressing noise in a signal.
class NoiseSuppressor {
 public:
  NoiseSuppressor(const NsConfig& config,
                  size_t sample_rate_hz,
                  size_t num_channels);
  NoiseSuppressor(const NoiseSuppressor&) = delete;
  NoiseSuppressor& operator=(const NoiseSuppressor&) = delete;

  // Analyses the signal (typically applied before the AEC to avoid analyzing
  // any comfort noise signal).
  void Analyze(const AudioBuffer& audio);

  // Applies noise suppression.
  void Process(AudioBuffer* audio);

  // Specifies whether the capture output will be used. The purpose of this is
  // to allow the noise suppressor to deactivate some of the processing when the
  // resulting output is anyway not used, for instance when the endpoint is
  // muted.
  void SetCaptureOutputUsage(bool capture_output_used) {
    capture_output_used_ = capture_output_used;
  }

 private:
  const size_t num_bands_;
  const size_t num_channels_;
  const SuppressionParams suppression_params_;
  int32_t num_analyzed_frames_ = -1;
  NrFft fft_;
  bool capture_output_used_ = true;

  struct ChannelState {
    ChannelState(const SuppressionParams& suppression_params, size_t num_bands);

    SpeechProbabilityEstimator speech_probability_estimator;
    WienerFilter wiener_filter;
    NoiseEstimator noise_estimator;
    std::array<float, kFftSizeBy2Plus1> prev_analysis_signal_spectrum;
    std::array<float, kFftSize - kNsFrameSize> analyze_analysis_memory;
    std::array<float, kOverlapSize> process_analysis_memory;
    std::array<float, kOverlapSize> process_synthesis_memory;
    std::vector<std::array<float, kOverlapSize>> process_delay_memory;
  };

  struct FilterBankState {
    std::array<float, kFftSize> real;
    std::array<float, kFftSize> imag;
    std::array<float, kFftSize> extended_frame;
  };

  std::vector<FilterBankState> filter_bank_states_heap_;
  std::vector<float> upper_band_gains_heap_;
  std::vector<float> energies_before_filtering_heap_;
  std::vector<float> gain_adjustments_heap_;
  std::vector<std::unique_ptr<ChannelState>> channels_;

  // Aggregates the Wiener filters into a single filter to use.
  void AggregateWienerFilters(
      rtc::ArrayView<float, kFftSizeBy2Plus1> filter) const;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_NOISE_SUPPRESSOR_H_
