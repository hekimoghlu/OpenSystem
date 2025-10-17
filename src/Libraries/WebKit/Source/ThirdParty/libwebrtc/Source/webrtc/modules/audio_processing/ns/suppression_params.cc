/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#include "modules/audio_processing/ns/suppression_params.h"

#include "rtc_base/checks.h"

namespace webrtc {

SuppressionParams::SuppressionParams(
    NsConfig::SuppressionLevel suppression_level) {
  switch (suppression_level) {
    case NsConfig::SuppressionLevel::k6dB:
      over_subtraction_factor = 1.f;
      // 6 dB attenuation.
      minimum_attenuating_gain = 0.5f;
      use_attenuation_adjustment = false;
      break;
    case NsConfig::SuppressionLevel::k12dB:
      over_subtraction_factor = 1.f;
      // 12 dB attenuation.
      minimum_attenuating_gain = 0.25f;
      use_attenuation_adjustment = true;
      break;
    case NsConfig::SuppressionLevel::k18dB:
      over_subtraction_factor = 1.1f;
      // 18 dB attenuation.
      minimum_attenuating_gain = 0.125f;
      use_attenuation_adjustment = true;
      break;
    case NsConfig::SuppressionLevel::k21dB:
      over_subtraction_factor = 1.25f;
      // 20.9 dB attenuation.
      minimum_attenuating_gain = 0.09f;
      use_attenuation_adjustment = true;
      break;
    default:
      RTC_DCHECK_NOTREACHED();
  }
}

}  // namespace webrtc
