/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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
#include "modules/desktop_capture/window_finder_mac.h"

#include <CoreFoundation/CoreFoundation.h>

#include <memory>
#include <utility>

#include "modules/desktop_capture/mac/desktop_configuration.h"
#include "modules/desktop_capture/mac/desktop_configuration_monitor.h"
#include "modules/desktop_capture/mac/window_list_utils.h"

namespace webrtc {

WindowFinderMac::WindowFinderMac(
    rtc::scoped_refptr<DesktopConfigurationMonitor> configuration_monitor)
    : configuration_monitor_(std::move(configuration_monitor)) {}
WindowFinderMac::~WindowFinderMac() = default;

WindowId WindowFinderMac::GetWindowUnderPoint(DesktopVector point) {
  WindowId id = kNullWindowId;
  GetWindowList(
      [&id, point](CFDictionaryRef window) {
        DesktopRect bounds;
        bounds = GetWindowBounds(window);
        if (bounds.Contains(point)) {
          id = GetWindowId(window);
          return false;
        }
        return true;
      },
      true,
      true);
  return id;
}

// static
std::unique_ptr<WindowFinder> WindowFinder::Create(
    const WindowFinder::Options& options) {
  return std::make_unique<WindowFinderMac>(options.configuration_monitor);
}

}  // namespace webrtc
