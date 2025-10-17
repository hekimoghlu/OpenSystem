/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
#include "rtc_base/experiments/keyframe_interval_settings.h"

#include "api/field_trials_view.h"

namespace webrtc {

namespace {

constexpr char kFieldTrialName[] = "WebRTC-KeyframeInterval";

}  // namespace

KeyframeIntervalSettings::KeyframeIntervalSettings(
    const FieldTrialsView& key_value_config)
    : min_keyframe_send_interval_ms_("min_keyframe_send_interval_ms") {
  ParseFieldTrial({&min_keyframe_send_interval_ms_},
                  key_value_config.Lookup(kFieldTrialName));
}

std::optional<int> KeyframeIntervalSettings::MinKeyframeSendIntervalMs() const {
  return min_keyframe_send_interval_ms_.GetOptional();
}
}  // namespace webrtc
