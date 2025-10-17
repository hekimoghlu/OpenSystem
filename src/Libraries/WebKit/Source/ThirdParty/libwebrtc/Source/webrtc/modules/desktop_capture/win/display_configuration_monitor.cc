/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "modules/desktop_capture/win/display_configuration_monitor.h"

#include <windows.h>

#include "modules/desktop_capture/win/screen_capture_utils.h"
#include "rtc_base/logging.h"

namespace webrtc {

bool DisplayConfigurationMonitor::IsChanged(
    DesktopCapturer::SourceId source_id) {
  DesktopRect rect = GetFullscreenRect();
  DesktopVector dpi = GetDpiForSourceId(source_id);

  if (!initialized_) {
    initialized_ = true;
    rect_ = rect;
    source_dpis_.emplace(source_id, std::move(dpi));
    return false;
  }

  if (!source_dpis_.contains(source_id)) {
    // If this is the first time we've seen this source_id, use the current DPI
    // so the monitor does not indicate a change and possibly get reset.
    source_dpis_.emplace(source_id, dpi);
  }

  bool has_changed = false;
  if (!rect.equals(rect_) || !source_dpis_.at(source_id).equals(dpi)) {
    has_changed = true;
    rect_ = rect;
    source_dpis_.emplace(source_id, std::move(dpi));
  }

  return has_changed;
}

void DisplayConfigurationMonitor::Reset() {
  initialized_ = false;
  source_dpis_.clear();
  rect_ = {};
}

DesktopVector DisplayConfigurationMonitor::GetDpiForSourceId(
    DesktopCapturer::SourceId source_id) {
  HMONITOR monitor = 0;
  if (source_id == kFullDesktopScreenId) {
    // Get a handle to the primary monitor when capturing the full desktop.
    monitor = MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY);
  } else if (!GetHmonitorFromDeviceIndex(source_id, &monitor)) {
    RTC_LOG(LS_WARNING) << "GetHmonitorFromDeviceIndex failed.";
  }
  return GetDpiForMonitor(monitor);
}

}  // namespace webrtc
