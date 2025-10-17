/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_HISTOGRAMS_H_
#define MODULES_AUDIO_PROCESSING_NS_HISTOGRAMS_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/ns/ns_common.h"
#include "modules/audio_processing/ns/signal_model.h"

namespace webrtc {

constexpr int kHistogramSize = 1000;

// Class for handling the updating of histograms.
class Histograms {
 public:
  Histograms();
  Histograms(const Histograms&) = delete;
  Histograms& operator=(const Histograms&) = delete;

  // Clears the histograms.
  void Clear();

  // Extracts thresholds for feature parameters and updates the corresponding
  // histogram.
  void Update(const SignalModel& features_);

  // Methods for accessing the histograms.
  rtc::ArrayView<const int, kHistogramSize> get_lrt() const { return lrt_; }
  rtc::ArrayView<const int, kHistogramSize> get_spectral_flatness() const {
    return spectral_flatness_;
  }
  rtc::ArrayView<const int, kHistogramSize> get_spectral_diff() const {
    return spectral_diff_;
  }

 private:
  std::array<int, kHistogramSize> lrt_;
  std::array<int, kHistogramSize> spectral_flatness_;
  std::array<int, kHistogramSize> spectral_diff_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_HISTOGRAMS_H_
