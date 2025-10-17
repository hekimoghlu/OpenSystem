/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
#ifndef MODULES_AUDIO_MIXER_DEFAULT_OUTPUT_RATE_CALCULATOR_H_
#define MODULES_AUDIO_MIXER_DEFAULT_OUTPUT_RATE_CALCULATOR_H_

#include <vector>

#include "api/array_view.h"
#include "modules/audio_mixer/output_rate_calculator.h"

namespace webrtc {

class DefaultOutputRateCalculator : public OutputRateCalculator {
 public:
  static const int kDefaultFrequency = 48000;

  // Produces the least native rate greater or equal to the preferred
  // sample rates. A native rate is one in
  // AudioProcessing::NativeRate. If `preferred_sample_rates` is
  // empty, returns `kDefaultFrequency`.
  int CalculateOutputRateFromRange(
      rtc::ArrayView<const int> preferred_sample_rates) override;
  ~DefaultOutputRateCalculator() override {}
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_DEFAULT_OUTPUT_RATE_CALCULATOR_H_
