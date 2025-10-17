/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_CLIPPING_PREDICTOR_H_
#define MODULES_AUDIO_PROCESSING_AGC2_CLIPPING_PREDICTOR_H_

#include <memory>
#include <optional>
#include <vector>

#include "api/audio/audio_processing.h"
#include "modules/audio_processing/include/audio_frame_view.h"

namespace webrtc {

// Frame-wise clipping prediction and clipped level step estimation. Analyzes
// 10 ms multi-channel frames and estimates an analog mic level decrease step
// to possibly avoid clipping when predicted. `Analyze()` and
// `EstimateClippedLevelStep()` can be called in any order.
class ClippingPredictor {
 public:
  virtual ~ClippingPredictor() = default;

  virtual void Reset() = 0;

  // Analyzes a 10 ms multi-channel audio frame.
  virtual void Analyze(const AudioFrameView<const float>& frame) = 0;

  // Predicts if clipping is going to occur for the specified `channel` in the
  // near-future and, if so, it returns a recommended analog mic level decrease
  // step. Returns std::nullopt if clipping is not predicted.
  // `level` is the current analog mic level, `default_step` is the amount the
  // mic level is lowered by the analog controller with every clipping event and
  // `min_mic_level` and `max_mic_level` is the range of allowed analog mic
  // levels.
  virtual std::optional<int> EstimateClippedLevelStep(
      int channel,
      int level,
      int default_step,
      int min_mic_level,
      int max_mic_level) const = 0;
};

// Creates a ClippingPredictor based on the provided `config`. When enabled,
// the following must hold for `config`:
// `window_length < reference_window_length + reference_window_delay`.
// Returns `nullptr` if `config.enabled` is false.
std::unique_ptr<ClippingPredictor> CreateClippingPredictor(
    int num_channels,
    const AudioProcessing::Config::GainController1::AnalogGainController::
        ClippingPredictor& config);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_CLIPPING_PREDICTOR_H_
