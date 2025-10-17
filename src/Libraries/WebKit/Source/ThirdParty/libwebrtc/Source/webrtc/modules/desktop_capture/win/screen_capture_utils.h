/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURE_UTILS_H_
#define MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURE_UTILS_H_

#if defined(WEBRTC_WIN)
// Forward declare HMONITOR in a windows.h compatible way so that we can avoid
// including windows.h.
#define WEBRTC_DECLARE_HANDLE(name) \
  struct name##__;                  \
  typedef struct name##__* name
WEBRTC_DECLARE_HANDLE(HMONITOR);
#undef WEBRTC_DECLARE_HANDLE
#endif

#include <string>
#include <vector>

#include "modules/desktop_capture/desktop_capturer.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Returns true if the system has at least one active display.
bool HasActiveDisplay();

// Output the list of active screens into `screens`. Returns true if succeeded,
// or false if it fails to enumerate the display devices. If the `device_names`
// is provided, it will be filled with the DISPLAY_DEVICE.DeviceName in UTF-8
// encoding. If this function returns true, consumers can always assume that
// `screens`[i] and `device_names`[i] indicate the same monitor on the system.
bool GetScreenList(DesktopCapturer::SourceList* screens,
                   std::vector<std::string>* device_names = nullptr);

// Converts a device index (which are returned by `GetScreenList`) into an
// HMONITOR.
bool GetHmonitorFromDeviceIndex(DesktopCapturer::SourceId device_index,
                                HMONITOR* hmonitor);

// Returns true if `monitor` represents a valid display
// monitor. Consumers should recheck the validity of HMONITORs before use if a
// WM_DISPLAYCHANGE message has been received.
bool IsMonitorValid(HMONITOR monitor);

// Returns the rect of the monitor identified by `monitor`, relative to the
// primary display's top-left. On failure, returns an empty rect.
DesktopRect GetMonitorRect(HMONITOR monitor);

// Returns the DPI for the specified monitor. On failure, returns the system DPI
// or the Windows default DPI (96x96) if the system DPI can't be retrieved.
DesktopVector GetDpiForMonitor(HMONITOR monitor);

// Returns true if `screen` is a valid screen. The screen device key is
// returned through `device_key` if the screen is valid. The device key can be
// used in GetScreenRect to verify the screen matches the previously obtained
// id.
bool IsScreenValid(DesktopCapturer::SourceId screen, std::wstring* device_key);

// Get the rect of the entire system in system coordinate system. I.e. the
// primary monitor always starts from (0, 0).
DesktopRect GetFullscreenRect();

// Get the rect of the screen identified by `screen`, relative to the primary
// display's top-left. If the screen device key does not match `device_key`, or
// the screen does not exist, or any error happens, an empty rect is returned.
RTC_EXPORT DesktopRect GetScreenRect(DesktopCapturer::SourceId screen,
                                     const std::wstring& device_key);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURE_UTILS_H_
