/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_FEATURES_EXTRACTION_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_FEATURES_EXTRACTION_H_

#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/agc2/biquad_filter.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/agc2/rnn_vad/pitch_search.h"
#include "modules/audio_processing/agc2/rnn_vad/sequence_buffer.h"
#include "modules/audio_processing/agc2/rnn_vad/spectral_features.h"

namespace webrtc {
namespace rnn_vad {

// Feature extractor to feed the VAD RNN.
class FeaturesExtractor {
 public:
  explicit FeaturesExtractor(const AvailableCpuFeatures& cpu_features);
  FeaturesExtractor(const FeaturesExtractor&) = delete;
  FeaturesExtractor& operator=(const FeaturesExtractor&) = delete;
  ~FeaturesExtractor();
  void Reset();
  // Analyzes the samples, computes the feature vector and returns true if
  // silence is detected (false if not). When silence is detected,
  // `feature_vector` is partially written and therefore must not be used to
  // feed the VAD RNN.
  bool CheckSilenceComputeFeatures(
      rtc::ArrayView<const float, kFrameSize10ms24kHz> samples,
      rtc::ArrayView<float, kFeatureVectorSize> feature_vector);

 private:
  const bool use_high_pass_filter_;
  // TODO(bugs.webrtc.org/7494): Remove HPF depending on how AGC2 is used in APM
  // and on whether an HPF is already used as pre-processing step in APM.
  BiQuadFilter hpf_;
  SequenceBuffer<float, kBufSize24kHz, kFrameSize10ms24kHz, kFrameSize20ms24kHz>
      pitch_buf_24kHz_;
  rtc::ArrayView<const float, kBufSize24kHz> pitch_buf_24kHz_view_;
  std::vector<float> lp_residual_;
  rtc::ArrayView<float, kBufSize24kHz> lp_residual_view_;
  PitchEstimator pitch_estimator_;
  rtc::ArrayView<const float, kFrameSize20ms24kHz> reference_frame_view_;
  SpectralFeaturesExtractor spectral_features_extractor_;
  int pitch_period_48kHz_;
};

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_FEATURES_EXTRACTION_H_
