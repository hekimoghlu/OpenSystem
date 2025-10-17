/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_MAC_H_
#define MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_MAC_H_

#include "api/scoped_refptr.h"
#include "modules/desktop_capture/window_finder.h"

namespace webrtc {

class DesktopConfigurationMonitor;

// The implementation of WindowFinder for Mac OSX.
class WindowFinderMac final : public WindowFinder {
 public:
  explicit WindowFinderMac(
      rtc::scoped_refptr<DesktopConfigurationMonitor> configuration_monitor);
  ~WindowFinderMac() override;

  // WindowFinder implementation.
  WindowId GetWindowUnderPoint(DesktopVector point) override;

 private:
  const rtc::scoped_refptr<DesktopConfigurationMonitor> configuration_monitor_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WINDOW_FINDER_MAC_H_
