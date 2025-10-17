/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_WIN_H_
#define MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_WIN_H_

#include "modules/desktop_capture/window_finder.h"

namespace webrtc {

// The implementation of WindowFinder for Windows.
class WindowFinderWin final : public WindowFinder {
 public:
  WindowFinderWin();
  ~WindowFinderWin() override;

  // WindowFinder implementation.
  WindowId GetWindowUnderPoint(DesktopVector point) override;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_WIN_H_
