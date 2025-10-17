/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#include "modules/desktop_capture/linux/x11/window_finder_x11.h"

#include <X11/X.h>

#include <memory>

#include "modules/desktop_capture/linux/x11/window_list_utils.h"
#include "rtc_base/checks.h"

namespace webrtc {

WindowFinderX11::WindowFinderX11(XAtomCache* cache) : cache_(cache) {
  RTC_DCHECK(cache_);
}

WindowFinderX11::~WindowFinderX11() = default;

WindowId WindowFinderX11::GetWindowUnderPoint(DesktopVector point) {
  WindowId id = kNullWindowId;
  GetWindowList(cache_, [&id, this, point](::Window window) {
    DesktopRect rect;
    if (GetWindowRect(this->cache_->display(), window, &rect) &&
        rect.Contains(point)) {
      id = window;
      return false;
    }
    return true;
  });
  return id;
}

// static
std::unique_ptr<WindowFinder> WindowFinder::Create(
    const WindowFinder::Options& options) {
  if (options.cache == nullptr) {
    return nullptr;
  }

  return std::make_unique<WindowFinderX11>(options.cache);
}

}  // namespace webrtc
