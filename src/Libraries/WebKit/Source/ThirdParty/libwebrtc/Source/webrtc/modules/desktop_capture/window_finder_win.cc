/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "modules/desktop_capture/window_finder_win.h"

#include <windows.h>

#include <memory>

namespace webrtc {

WindowFinderWin::WindowFinderWin() = default;
WindowFinderWin::~WindowFinderWin() = default;

WindowId WindowFinderWin::GetWindowUnderPoint(DesktopVector point) {
  HWND window = WindowFromPoint(POINT{point.x(), point.y()});
  if (!window) {
    return kNullWindowId;
  }

  // The difference between GA_ROOTOWNER and GA_ROOT can be found at
  // https://groups.google.com/a/chromium.org/forum/#!topic/chromium-dev/Hirr_DkuZdw.
  // In short, we should use GA_ROOT, since we only care about the root window
  // but not the owner.
  window = GetAncestor(window, GA_ROOT);
  if (!window) {
    return kNullWindowId;
  }

  return reinterpret_cast<WindowId>(window);
}

// static
std::unique_ptr<WindowFinder> WindowFinder::Create(
    const WindowFinder::Options& options) {
  return std::make_unique<WindowFinderWin>();
}

}  // namespace webrtc
