/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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
#include "api/video/video_adaptation_counters.h"

#include <string>

#include "rtc_base/strings/string_builder.h"

namespace webrtc {

bool VideoAdaptationCounters::operator==(
    const VideoAdaptationCounters& rhs) const {
  return fps_adaptations == rhs.fps_adaptations &&
         resolution_adaptations == rhs.resolution_adaptations;
}

bool VideoAdaptationCounters::operator!=(
    const VideoAdaptationCounters& rhs) const {
  return !(rhs == *this);
}

VideoAdaptationCounters VideoAdaptationCounters::operator+(
    const VideoAdaptationCounters& other) const {
  return VideoAdaptationCounters(
      resolution_adaptations + other.resolution_adaptations,
      fps_adaptations + other.fps_adaptations);
}

std::string VideoAdaptationCounters::ToString() const {
  rtc::StringBuilder ss;
  ss << "{ res=" << resolution_adaptations << " fps=" << fps_adaptations
     << " }";
  return ss.Release();
}

}  // namespace webrtc
