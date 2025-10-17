/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#ifndef MODULES_VIDEO_CODING_SVC_CREATE_SCALABILITY_STRUCTURE_H_
#define MODULES_VIDEO_CODING_SVC_CREATE_SCALABILITY_STRUCTURE_H_

#include <memory>
#include <optional>
#include <vector>

#include "api/video_codecs/scalability_mode.h"
#include "modules/video_coding/svc/scalable_video_controller.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Creates a structure by name according to
// https://w3c.github.io/webrtc-svc/#scalabilitymodes*
// Returns nullptr for unknown name.
std::unique_ptr<ScalableVideoController> RTC_EXPORT
CreateScalabilityStructure(ScalabilityMode name);

// Returns description of the scalability structure identified by 'name',
// Return nullopt for unknown name.
std::optional<ScalableVideoController::StreamLayersConfig>
ScalabilityStructureConfig(ScalabilityMode name);

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_SVC_CREATE_SCALABILITY_STRUCTURE_H_
