/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "modules/audio_processing/include/aec_dump.h"

namespace webrtc {
InternalAPMConfig::InternalAPMConfig() = default;
InternalAPMConfig::InternalAPMConfig(const InternalAPMConfig&) = default;
InternalAPMConfig::InternalAPMConfig(InternalAPMConfig&&) = default;
InternalAPMConfig& InternalAPMConfig::operator=(const InternalAPMConfig&) =
    default;

bool InternalAPMConfig::operator==(const InternalAPMConfig& other) const {
  return aec_enabled == other.aec_enabled &&
         aec_delay_agnostic_enabled == other.aec_delay_agnostic_enabled &&
         aec_drift_compensation_enabled ==
             other.aec_drift_compensation_enabled &&
         aec_extended_filter_enabled == other.aec_extended_filter_enabled &&
         aec_suppression_level == other.aec_suppression_level &&
         aecm_enabled == other.aecm_enabled &&
         aecm_comfort_noise_enabled == other.aecm_comfort_noise_enabled &&
         aecm_routing_mode == other.aecm_routing_mode &&
         agc_enabled == other.agc_enabled && agc_mode == other.agc_mode &&
         agc_limiter_enabled == other.agc_limiter_enabled &&
         hpf_enabled == other.hpf_enabled && ns_enabled == other.ns_enabled &&
         ns_level == other.ns_level &&
         transient_suppression_enabled == other.transient_suppression_enabled &&
         noise_robust_agc_enabled == other.noise_robust_agc_enabled &&
         pre_amplifier_enabled == other.pre_amplifier_enabled &&
         pre_amplifier_fixed_gain_factor ==
             other.pre_amplifier_fixed_gain_factor &&
         experiments_description == other.experiments_description;
}
}  // namespace webrtc
