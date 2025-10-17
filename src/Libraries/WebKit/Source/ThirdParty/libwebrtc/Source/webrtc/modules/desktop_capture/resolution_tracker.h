/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_RESOLUTION_TRACKER_H_
#define MODULES_DESKTOP_CAPTURE_RESOLUTION_TRACKER_H_

#include "modules/desktop_capture/desktop_geometry.h"

namespace webrtc {

class ResolutionTracker final {
 public:
  // Sets the resolution to `size`. Returns true if a previous size was recorded
  // and differs from `size`.
  bool SetResolution(DesktopSize size);

  // Resets to the initial state.
  void Reset();

 private:
  DesktopSize last_size_;
  bool initialized_ = false;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_RESOLUTION_TRACKER_H_
