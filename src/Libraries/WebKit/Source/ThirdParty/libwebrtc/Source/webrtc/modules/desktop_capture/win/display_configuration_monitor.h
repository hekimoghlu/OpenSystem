/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_DISPLAY_CONFIGURATION_MONITOR_H_
#define MODULES_DESKTOP_CAPTURE_WIN_DISPLAY_CONFIGURATION_MONITOR_H_

#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "rtc_base/containers/flat_map.h"

namespace webrtc {

// A passive monitor to detect the change of display configuration on a Windows
// system.
// TODO(zijiehe): Also check for pixel format changes.
class DisplayConfigurationMonitor {
 public:
  // Checks whether the display configuration has changed since the last time
  // IsChanged() was called. |source_id| is used to observe changes for a
  // specific display or all displays if kFullDesktopScreenId is passed in.
  // Returns false if object was Reset() or if IsChanged() has not been called.
  bool IsChanged(DesktopCapturer::SourceId source_id);

  // Resets to the initial state.
  void Reset();

 private:
  DesktopVector GetDpiForSourceId(DesktopCapturer::SourceId source_id);

  // Represents the size of the desktop which includes all displays.
  DesktopRect rect_;

  // Tracks the DPI for each display being captured. We need to track for each
  // display as each one can be configured to use a different DPI which will not
  // be reflected in calls to get the system DPI.
  flat_map<DesktopCapturer::SourceId, DesktopVector> source_dpis_;

  // Indicates whether |rect_| and |source_dpis_| have been initialized. This is
  // used to prevent the monitor instance from signaling 'IsChanged()' before
  // the initial values have been set.
  bool initialized_ = false;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_DISPLAY_CONFIGURATION_MONITOR_H_
