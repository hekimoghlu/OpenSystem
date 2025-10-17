/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_VAD_VAD_AUDIO_PROC_H_
#define MODULES_AUDIO_PROCESSING_VAD_VAD_AUDIO_PROC_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "modules/audio_processing/vad/common.h"  // AudioFeatures, kSampleR...

namespace webrtc {

class PoleZeroFilter;

class VadAudioProc {
 public:
  // Forward declare iSAC structs.
  struct PitchAnalysisStruct;
  struct PreFiltBankstr;

  VadAudioProc();
  ~VadAudioProc();

  int ExtractFeatures(const int16_t* audio_frame,
                      size_t length,
                      AudioFeatures* audio_features);

  static constexpr size_t kDftSize = 512;

 private:
  void PitchAnalysis(double* pitch_gains, double* pitch_lags_hz, size_t length);
  void SubframeCorrelation(double* corr,
                           size_t length_corr,
                           size_t subframe_index);
  void GetLpcPolynomials(double* lpc, size_t length_lpc);
  void FindFirstSpectralPeaks(double* f_peak, size_t length_f_peak);
  void Rms(double* rms, size_t length_rms);
  void ResetBuffer();

  // To compute spectral peak we perform LPC analysis to get spectral envelope.
  // For every 30 ms we compute 3 spectral peak there for 3 LPC analysis.
  // LPC is computed over 15 ms of windowed audio. For every 10 ms sub-frame
  // we need 5 ms of past signal to create the input of LPC analysis.
  static constexpr size_t kNumPastSignalSamples = size_t{kSampleRateHz / 200};

  // TODO(turajs): maybe defining this at a higher level (maybe enum) so that
  // all the code recognize it as "no-error."
  static constexpr int kNoError = 0;

  static constexpr size_t kNum10msSubframes = 3;
  static constexpr size_t kNumSubframeSamples = size_t{kSampleRateHz / 100};
  // Samples in 30 ms @ given sampling rate.
  static constexpr size_t kNumSamplesToProcess =
      kNum10msSubframes * kNumSubframeSamples;
  static constexpr size_t kBufferLength =
      kNumPastSignalSamples + kNumSamplesToProcess;
  static constexpr size_t kIpLength = kDftSize >> 1;
  static constexpr size_t kWLength = kDftSize >> 1;
  static constexpr size_t kLpcOrder = 16;

  size_t ip_[kIpLength];
  float w_fft_[kWLength];

  // A buffer of 5 ms (past audio) + 30 ms (one iSAC frame ).
  float audio_buffer_[kBufferLength];
  size_t num_buffer_samples_;

  double log_old_gain_;
  double old_lag_;

  std::unique_ptr<PitchAnalysisStruct> pitch_analysis_handle_;
  std::unique_ptr<PreFiltBankstr> pre_filter_handle_;
  std::unique_ptr<PoleZeroFilter> high_pass_filter_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_VAD_VAD_AUDIO_PROC_H_
