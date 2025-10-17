/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#ifndef MODULES_AUDIO_MIXER_OUTPUT_RATE_CALCULATOR_H_
#define MODULES_AUDIO_MIXER_OUTPUT_RATE_CALCULATOR_H_

#include <vector>

#include "api/array_view.h"

namespace webrtc {

// Decides the sample rate of a mixing iteration given the preferred
// sample rates of the sources.
class OutputRateCalculator {
 public:
  virtual int CalculateOutputRateFromRange(
      rtc::ArrayView<const int> preferred_sample_rates) = 0;

  virtual ~OutputRateCalculator() {}
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_OUTPUT_RATE_CALCULATOR_H_
