/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_H_
#define MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_H_

#include <memory>

#include "api/scoped_refptr.h"
#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_geometry.h"

#if defined(WEBRTC_MAC) && !defined(WEBRTC_IOS)
#include "modules/desktop_capture/mac/desktop_configuration_monitor.h"
#endif

namespace webrtc {

#if defined(WEBRTC_USE_X11)
class XAtomCache;
#endif

// An interface to return the id of the visible window under a certain point.
class WindowFinder {
 public:
  WindowFinder() = default;
  virtual ~WindowFinder() = default;

  // Returns the id of the visible window under `point`. This function returns
  // kNullWindowId if no window is under `point` and the platform does not have
  // "root window" concept, i.e. the visible area under `point` is the desktop.
  // `point` is always in system coordinate, i.e. the primary monitor always
  // starts from (0, 0).
  virtual WindowId GetWindowUnderPoint(DesktopVector point) = 0;

  struct Options final {
    Options();
    ~Options();
    Options(const Options& other);
    Options(Options&& other);

#if defined(WEBRTC_USE_X11)
    XAtomCache* cache = nullptr;
#endif
#if defined(WEBRTC_MAC) && !defined(WEBRTC_IOS)
    rtc::scoped_refptr<DesktopConfigurationMonitor> configuration_monitor;
#endif
  };

  // Creates a platform-independent WindowFinder implementation. This function
  // returns nullptr if `options` does not contain enough information or
  // WindowFinder does not support current platform.
  static std::unique_ptr<WindowFinder> Create(const Options& options);
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_H_
