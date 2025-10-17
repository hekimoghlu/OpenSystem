/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_SUBTRACTOR_OUTPUT_ANALYZER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_SUBTRACTOR_OUTPUT_ANALYZER_H_

#include <vector>

#include "modules/audio_processing/aec3/subtractor_output.h"

namespace webrtc {

// Class for analyzing the properties subtractor output.
class SubtractorOutputAnalyzer {
 public:
  explicit SubtractorOutputAnalyzer(size_t num_capture_channels);
  ~SubtractorOutputAnalyzer() = default;

  // Analyses the subtractor output.
  void Update(rtc::ArrayView<const SubtractorOutput> subtractor_output,
              bool* any_filter_converged,
              bool* any_coarse_filter_converged,
              bool* all_filters_diverged);

  const std::vector<bool>& ConvergedFilters() const {
    return filters_converged_;
  }

  // Handle echo path change.
  void HandleEchoPathChange();

 private:
  std::vector<bool> filters_converged_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_SUBTRACTOR_OUTPUT_ANALYZER_H_
