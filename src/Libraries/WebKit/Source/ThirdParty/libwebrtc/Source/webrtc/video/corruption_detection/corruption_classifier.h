/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#ifndef VIDEO_CORRUPTION_DETECTION_CORRUPTION_CLASSIFIER_H_
#define VIDEO_CORRUPTION_DETECTION_CORRUPTION_CLASSIFIER_H_

#include <variant>

#include "api/array_view.h"
#include "video/corruption_detection/halton_frame_sampler.h"

namespace webrtc {

// Based on the given filtered samples to `CalculateCorruptionProbability` this
// class calculates a probability to indicate whether the frame is corrupted.
// The classification is done either by scaling the loss to the interval of [0,
// 1] using a simple `scale_factor` or by applying a logistic function to the
// loss. The logistic function is constructed based on `growth_rate` and
// `midpoint`, to the score between the original and the compressed frames'
// samples. This score is calculated using `GetScore`.
//
// TODO: bugs.webrtc.org/358039777 - Remove one of the constructors based on
// which mapping function works best in practice.
class CorruptionClassifier {
 public:
  // Calculates the corruption probability using a simple scale factor.
  explicit CorruptionClassifier(float scale_factor);
  // Calculates the corruption probability using a logistic function.
  CorruptionClassifier(float growth_rate, float midpoint);
  ~CorruptionClassifier() = default;

  // This function calculates and returns the probability (in the interval [0,
  // 1] that a frame is corrupted. The probability is determined either by
  // scaling the loss to the interval of [0, 1] using a simple `scale_factor`
  // or by applying a logistic function to the loss. The method is chosen
  // depending on the used constructor.
  double CalculateCorruptionProbability(
      rtc::ArrayView<const FilteredSample> filtered_original_samples,
      rtc::ArrayView<const FilteredSample> filtered_compressed_samples,
      int luma_threshold,
      int chroma_threshold) const;

 private:
  struct ScalarConfig {
    float scale_factor;
  };

  // Logistic function parameters. See
  // https://en.wikipedia.org/wiki/Logistic_function.
  struct LogisticFunctionConfig {
    float growth_rate;
    float midpoint;
  };

  // Returns the non-normalized score between the original and the compressed
  // frames' samples.
  double GetScore(
      rtc::ArrayView<const FilteredSample> filtered_original_samples,
      rtc::ArrayView<const FilteredSample> filtered_compressed_samples,
      int luma_threshold,
      int chroma_threshold) const;

  const std::variant<ScalarConfig, LogisticFunctionConfig> config_;
};

}  // namespace webrtc

#endif  // VIDEO_CORRUPTION_DETECTION_CORRUPTION_CLASSIFIER_H_
