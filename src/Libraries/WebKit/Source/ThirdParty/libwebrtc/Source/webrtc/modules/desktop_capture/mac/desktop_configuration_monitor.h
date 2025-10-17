/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_CONFIGURATION_MONITOR_H_
#define MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_CONFIGURATION_MONITOR_H_

#include <ApplicationServices/ApplicationServices.h>

#include <memory>
#include <set>

#include "api/ref_counted_base.h"
#include "modules/desktop_capture/mac/desktop_configuration.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

// The class provides functions to synchronize capturing and display
// reconfiguring across threads, and the up-to-date MacDesktopConfiguration.
class DesktopConfigurationMonitor final
    : public rtc::RefCountedNonVirtual<DesktopConfigurationMonitor> {
 public:
  DesktopConfigurationMonitor();
  ~DesktopConfigurationMonitor();

  DesktopConfigurationMonitor(const DesktopConfigurationMonitor&) = delete;
  DesktopConfigurationMonitor& operator=(const DesktopConfigurationMonitor&) =
      delete;

  // Returns the current desktop configuration.
  MacDesktopConfiguration desktop_configuration();

 private:
  static void DisplaysReconfiguredCallback(CGDirectDisplayID display,
                                           CGDisplayChangeSummaryFlags flags,
                                           void* user_parameter);
  void DisplaysReconfigured(CGDirectDisplayID display,
                            CGDisplayChangeSummaryFlags flags);

  Mutex desktop_configuration_lock_;
  MacDesktopConfiguration desktop_configuration_
      RTC_GUARDED_BY(&desktop_configuration_lock_);
  std::set<CGDirectDisplayID> reconfiguring_displays_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_CONFIGURATION_MONITOR_H_
