/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_RENDER_SIGNAL_ANALYZER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_RENDER_SIGNAL_ANALYZER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Provides functionality for analyzing the properties of the render signal.
class RenderSignalAnalyzer {
 public:
  explicit RenderSignalAnalyzer(const EchoCanceller3Config& config);
  ~RenderSignalAnalyzer();

  RenderSignalAnalyzer(const RenderSignalAnalyzer&) = delete;
  RenderSignalAnalyzer& operator=(const RenderSignalAnalyzer&) = delete;

  // Updates the render signal analysis with the most recent render signal.
  void Update(const RenderBuffer& render_buffer,
              const std::optional<size_t>& delay_partitions);

  // Returns true if the render signal is poorly exciting.
  bool PoorSignalExcitation() const {
    RTC_DCHECK_LT(2, narrow_band_counters_.size());
    return std::any_of(narrow_band_counters_.begin(),
                       narrow_band_counters_.end(),
                       [](size_t a) { return a > 10; });
  }

  // Zeros the array around regions with narrow bands signal characteristics.
  void MaskRegionsAroundNarrowBands(
      std::array<float, kFftLengthBy2Plus1>* v) const;

  std::optional<int> NarrowPeakBand() const { return narrow_peak_band_; }

 private:
  const int strong_peak_freeze_duration_;
  std::array<size_t, kFftLengthBy2 - 1> narrow_band_counters_;
  std::optional<int> narrow_peak_band_;
  size_t narrow_peak_counter_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_RENDER_SIGNAL_ANALYZER_H_
