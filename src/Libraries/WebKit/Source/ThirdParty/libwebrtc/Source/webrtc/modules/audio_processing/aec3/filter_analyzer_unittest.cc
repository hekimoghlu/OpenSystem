/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include "modules/audio_processing/aec3/filter_analyzer.h"

#include <algorithm>

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

// Verifies that the filter analyzer handles filter resizes properly.
TEST(FilterAnalyzer, FilterResize) {
  EchoCanceller3Config c;
  std::vector<float> filter(65, 0.f);
  for (size_t num_capture_channels : {1, 2, 4}) {
    FilterAnalyzer fa(c, num_capture_channels);
    fa.SetRegionToAnalyze(filter.size());
    fa.SetRegionToAnalyze(filter.size());
    filter.resize(32);
    fa.SetRegionToAnalyze(filter.size());
  }
}

}  // namespace webrtc
