/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#include "modules/desktop_capture/mac/desktop_configuration_monitor.h"

#include "modules/desktop_capture/mac/desktop_configuration.h"
#include "rtc_base/logging.h"
#include "rtc_base/trace_event.h"

namespace webrtc {

DesktopConfigurationMonitor::DesktopConfigurationMonitor() {
  CGError err = CGDisplayRegisterReconfigurationCallback(
      DesktopConfigurationMonitor::DisplaysReconfiguredCallback, this);
  if (err != kCGErrorSuccess)
    RTC_LOG(LS_ERROR) << "CGDisplayRegisterReconfigurationCallback " << err;
  MutexLock lock(&desktop_configuration_lock_);
  desktop_configuration_ = MacDesktopConfiguration::GetCurrent(
      MacDesktopConfiguration::TopLeftOrigin);
}

DesktopConfigurationMonitor::~DesktopConfigurationMonitor() {
  CGError err = CGDisplayRemoveReconfigurationCallback(
      DesktopConfigurationMonitor::DisplaysReconfiguredCallback, this);
  if (err != kCGErrorSuccess)
    RTC_LOG(LS_ERROR) << "CGDisplayRemoveReconfigurationCallback " << err;
}

MacDesktopConfiguration DesktopConfigurationMonitor::desktop_configuration() {
  MutexLock lock(&desktop_configuration_lock_);
  return desktop_configuration_;
}

// static
// This method may be called on any system thread.
void DesktopConfigurationMonitor::DisplaysReconfiguredCallback(
    CGDirectDisplayID display,
    CGDisplayChangeSummaryFlags flags,
    void* user_parameter) {
  DesktopConfigurationMonitor* monitor =
      reinterpret_cast<DesktopConfigurationMonitor*>(user_parameter);
  monitor->DisplaysReconfigured(display, flags);
}

void DesktopConfigurationMonitor::DisplaysReconfigured(
    CGDirectDisplayID display,
    CGDisplayChangeSummaryFlags flags) {
  TRACE_EVENT0("webrtc", "DesktopConfigurationMonitor::DisplaysReconfigured");
  RTC_LOG(LS_INFO) << "DisplaysReconfigured: "
                      "DisplayID "
                   << display << "; ChangeSummaryFlags " << flags;

  if (flags & kCGDisplayBeginConfigurationFlag) {
    reconfiguring_displays_.insert(display);
    return;
  }

  reconfiguring_displays_.erase(display);
  if (reconfiguring_displays_.empty()) {
    MutexLock lock(&desktop_configuration_lock_);
    desktop_configuration_ = MacDesktopConfiguration::GetCurrent(
        MacDesktopConfiguration::TopLeftOrigin);
  }
}

}  // namespace webrtc
