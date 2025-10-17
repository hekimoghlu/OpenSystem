/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#ifndef API_VIDEO_CODECS_SCALABILITY_MODE_HELPER_H_
#define API_VIDEO_CODECS_SCALABILITY_MODE_HELPER_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "api/video_codecs/scalability_mode.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Returns the number of spatial layers from the `scalability_mode_string`
// or nullopt if the given mode is unknown.
RTC_EXPORT std::optional<int> ScalabilityModeStringToNumSpatialLayers(
    absl::string_view scalability_mode_string);

// Returns the number of temporal layers from the `scalability_mode_string`
// or nullopt if the given mode is unknown.
RTC_EXPORT std::optional<int> ScalabilityModeStringToNumTemporalLayers(
    absl::string_view scalability_mode_string);

// Convert the `scalability_mode_string` to the scalability mode enum value
// or nullopt if the given mode is unknown.
RTC_EXPORT std::optional<ScalabilityMode> ScalabilityModeStringToEnum(
    absl::string_view scalability_mode_string);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_SCALABILITY_MODE_HELPER_H_
