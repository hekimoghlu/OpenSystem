/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
#include "modules/audio_mixer/default_output_rate_calculator.h"

#include <algorithm>
#include <iterator>

#include "api/audio/audio_processing.h"
#include "rtc_base/checks.h"

namespace webrtc {

int DefaultOutputRateCalculator::CalculateOutputRateFromRange(
    rtc::ArrayView<const int> preferred_sample_rates) {
  if (preferred_sample_rates.empty()) {
    return DefaultOutputRateCalculator::kDefaultFrequency;
  }
  using NativeRate = AudioProcessing::NativeRate;
  const int maximal_frequency = *std::max_element(
      preferred_sample_rates.cbegin(), preferred_sample_rates.cend());

  RTC_DCHECK_LE(NativeRate::kSampleRate8kHz, maximal_frequency);
  RTC_DCHECK_GE(NativeRate::kSampleRate48kHz, maximal_frequency);

  static constexpr NativeRate native_rates[] = {
      NativeRate::kSampleRate8kHz, NativeRate::kSampleRate16kHz,
      NativeRate::kSampleRate32kHz, NativeRate::kSampleRate48kHz};
  const auto* rounded_up_index = std::lower_bound(
      std::begin(native_rates), std::end(native_rates), maximal_frequency);
  RTC_DCHECK(rounded_up_index != std::end(native_rates));
  return *rounded_up_index;
}
}  // namespace webrtc
