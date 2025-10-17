/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_
#define MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_

#include "modules/audio_processing/ns/ns_config.h"

namespace webrtc {

struct SuppressionParams {
  explicit SuppressionParams(NsConfig::SuppressionLevel suppression_level);
  SuppressionParams(const SuppressionParams&) = delete;
  SuppressionParams& operator=(const SuppressionParams&) = delete;

  float over_subtraction_factor;
  float minimum_attenuating_gain;
  bool use_attenuation_adjustment;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_SUPPRESSION_PARAMS_H_
