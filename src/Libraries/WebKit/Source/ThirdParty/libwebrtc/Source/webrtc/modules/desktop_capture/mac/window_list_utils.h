/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_WINDOW_LIST_UTILS_H_
#define MODULES_DESKTOP_CAPTURE_MAC_WINDOW_LIST_UTILS_H_

#include <ApplicationServices/ApplicationServices.h>

#include <string>

#include "api/function_view.h"
#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/mac/desktop_configuration.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Iterates all on-screen windows in decreasing z-order and sends them
// one-by-one to `on_window` function. If `on_window` returns false, this
// function returns immediately. GetWindowList() returns false if native APIs
// failed. Menus, dock (if `only_zero_layer`), minimized windows (if
// `ignore_minimized` is true) and any windows which do not have a valid window
// id or title will be ignored.
bool RTC_EXPORT
GetWindowList(rtc::FunctionView<bool(CFDictionaryRef)> on_window,
              bool ignore_minimized,
              bool only_zero_layer);

// Another helper function to get the on-screen windows.
bool RTC_EXPORT GetWindowList(DesktopCapturer::SourceList* windows,
                              bool ignore_minimized,
                              bool only_zero_layer);

// Returns true if the window is occupying a full screen.
bool IsWindowFullScreen(const MacDesktopConfiguration& desktop_config,
                        CFDictionaryRef window);

// Returns true if the window is occupying a full screen.
bool IsWindowFullScreen(const MacDesktopConfiguration& desktop_config,
                        CGWindowID id);

// Returns true if the `window` is on screen. This function returns false if
// native APIs fail.
bool IsWindowOnScreen(CFDictionaryRef window);

// Returns true if the window is on screen. This function returns false if
// native APIs fail or `id` cannot be found.
bool IsWindowOnScreen(CGWindowID id);

// Returns utf-8 encoded title of `window`. If `window` is not a window or no
// valid title can be retrieved, this function returns an empty string.
std::string GetWindowTitle(CFDictionaryRef window);

// Returns utf-8 encoded title of window `id`. If `id` cannot be found or no
// valid title can be retrieved, this function returns an empty string.
std::string GetWindowTitle(CGWindowID id);

// Returns utf-8 encoded owner name of `window`. If `window` is not a window or
// if no valid owner name can be retrieved, returns an empty string.
std::string GetWindowOwnerName(CFDictionaryRef window);

// Returns utf-8 encoded owner name of the given window `id`. If `id` cannot be
// found or if no valid owner name can be retrieved, returns an empty string.
std::string GetWindowOwnerName(CGWindowID id);

// Returns id of `window`. If `window` is not a window or the window id cannot
// be retrieved, this function returns kNullWindowId.
WindowId GetWindowId(CFDictionaryRef window);

// Returns the pid of the process owning `window`. Return 0 if `window` is not
// a window or no valid owner can be retrieved.
int GetWindowOwnerPid(CFDictionaryRef window);

// Returns the pid of the process owning the window `id`. Return 0 if `id`
// cannot be found or no valid owner can be retrieved.
int GetWindowOwnerPid(CGWindowID id);

// Returns the DIP to physical pixel scale at `position`. `position` is in
// *unscaled* system coordinate, i.e. it's device-independent and the primary
// monitor starts from (0, 0). If `position` is out of the system display, this
// function returns 1.
float GetScaleFactorAtPosition(const MacDesktopConfiguration& desktop_config,
                               DesktopVector position);

// Returns the DIP to physical pixel scale factor of the window with `id`.
// The bounds of the window with `id` is in DIP coordinates and `size` is the
// CGImage size of the window with `id` in physical coordinates. Comparing them
// can give the current scale factor.
// If the window overlaps multiple monitors, OS will decide on which monitor the
// window is displayed and use its scale factor to the window. So this method
// still works.
float GetWindowScaleFactor(CGWindowID id, DesktopSize size);

// Returns the bounds of `window`. If `window` is not a window or the bounds
// cannot be retrieved, this function returns an empty DesktopRect. The returned
// DesktopRect is in system coordinate, i.e. the primary monitor always starts
// from (0, 0).
// Deprecated: This function should be avoided in favor of the overload with
// MacDesktopConfiguration.
DesktopRect GetWindowBounds(CFDictionaryRef window);

// Returns the bounds of window with `id`. If `id` does not represent a window
// or the bounds cannot be retrieved, this function returns an empty
// DesktopRect. The returned DesktopRect is in system coordinates.
// Deprecated: This function should be avoided in favor of the overload with
// MacDesktopConfiguration.
DesktopRect GetWindowBounds(CGWindowID id);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_WINDOW_LIST_UTILS_H_
