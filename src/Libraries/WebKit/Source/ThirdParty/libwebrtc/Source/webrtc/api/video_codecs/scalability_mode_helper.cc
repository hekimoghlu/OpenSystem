/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#include "api/video_codecs/scalability_mode_helper.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "api/video_codecs/scalability_mode.h"
#include "modules/video_coding/svc/scalability_mode_util.h"

namespace webrtc {

std::optional<int> ScalabilityModeStringToNumSpatialLayers(
    absl::string_view scalability_mode_string) {
  std::optional<ScalabilityMode> scalability_mode =
      ScalabilityModeFromString(scalability_mode_string);
  if (!scalability_mode.has_value()) {
    return std::nullopt;
  }
  return ScalabilityModeToNumSpatialLayers(*scalability_mode);
}

std::optional<int> ScalabilityModeStringToNumTemporalLayers(
    absl::string_view scalability_mode_string) {
  std::optional<ScalabilityMode> scalability_mode =
      ScalabilityModeFromString(scalability_mode_string);
  if (!scalability_mode.has_value()) {
    return std::nullopt;
  }
  return ScalabilityModeToNumTemporalLayers(*scalability_mode);
}

std::optional<ScalabilityMode> ScalabilityModeStringToEnum(
    absl::string_view scalability_mode_string) {
  return ScalabilityModeFromString(scalability_mode_string);
}

}  // namespace webrtc
