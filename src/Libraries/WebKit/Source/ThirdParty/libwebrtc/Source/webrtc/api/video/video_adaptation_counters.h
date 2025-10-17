/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#ifndef API_VIDEO_VIDEO_ADAPTATION_COUNTERS_H_
#define API_VIDEO_VIDEO_ADAPTATION_COUNTERS_H_

#include <string>

#include "rtc_base/checks.h"

namespace webrtc {

// Counts the number of adaptations have resulted due to resource overuse.
// Today we can adapt resolution and fps.
struct VideoAdaptationCounters {
  VideoAdaptationCounters() : resolution_adaptations(0), fps_adaptations(0) {}
  VideoAdaptationCounters(int resolution_adaptations, int fps_adaptations)
      : resolution_adaptations(resolution_adaptations),
        fps_adaptations(fps_adaptations) {
    RTC_DCHECK_GE(resolution_adaptations, 0);
    RTC_DCHECK_GE(fps_adaptations, 0);
  }

  int Total() const { return fps_adaptations + resolution_adaptations; }

  bool operator==(const VideoAdaptationCounters& rhs) const;
  bool operator!=(const VideoAdaptationCounters& rhs) const;

  VideoAdaptationCounters operator+(const VideoAdaptationCounters& other) const;

  std::string ToString() const;

  int resolution_adaptations;
  int fps_adaptations;
};

}  // namespace webrtc

#endif  // API_VIDEO_VIDEO_ADAPTATION_COUNTERS_H_
