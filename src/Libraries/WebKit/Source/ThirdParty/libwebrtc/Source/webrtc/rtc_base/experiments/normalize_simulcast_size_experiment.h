/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
#ifndef RTC_BASE_EXPERIMENTS_NORMALIZE_SIMULCAST_SIZE_EXPERIMENT_H_
#define RTC_BASE_EXPERIMENTS_NORMALIZE_SIMULCAST_SIZE_EXPERIMENT_H_

#include <optional>

#include "api/field_trials_view.h"

namespace webrtc {
class NormalizeSimulcastSizeExperiment {
 public:
  // Returns the base two exponent from field trial.
  static std::optional<int> GetBase2Exponent(
      const FieldTrialsView& field_trials);
};

}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_NORMALIZE_SIMULCAST_SIZE_EXPERIMENT_H_
