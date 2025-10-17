/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "rtc_base/experiments/normalize_simulcast_size_experiment.h"

#include <stdio.h>

#include <string>

#include "api/field_trials_view.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {
constexpr char kFieldTrial[] = "WebRTC-NormalizeSimulcastResolution";
constexpr int kMinSetting = 0;
constexpr int kMaxSetting = 5;
}  // namespace

std::optional<int> NormalizeSimulcastSizeExperiment::GetBase2Exponent(
    const FieldTrialsView& field_trials) {
  if (!field_trials.IsEnabled(kFieldTrial))
    return std::nullopt;

  const std::string group = field_trials.Lookup(kFieldTrial);
  if (group.empty())
    return std::nullopt;

  int exponent;
  if (sscanf(group.c_str(), "Enabled-%d", &exponent) != 1) {
    RTC_LOG(LS_WARNING) << "No parameter provided.";
    return std::nullopt;
  }

  if (exponent < kMinSetting || exponent > kMaxSetting) {
    RTC_LOG(LS_WARNING) << "Unsupported exp value provided, value ignored.";
    return std::nullopt;
  }

  return std::optional<int>(exponent);
}

}  // namespace webrtc
