/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include "modules/desktop_capture/desktop_capture_options.h"

#include "api/make_ref_counted.h"

#if defined(WEBRTC_MAC) && !defined(WEBRTC_IOS)
#include "modules/desktop_capture/mac/full_screen_mac_application_handler.h"
#elif defined(WEBRTC_WIN)
#include "modules/desktop_capture/win/full_screen_win_application_handler.h"
#endif
#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/desktop_capture/linux/wayland/shared_screencast_stream.h"
#endif

namespace webrtc {

DesktopCaptureOptions::DesktopCaptureOptions() {}
DesktopCaptureOptions::DesktopCaptureOptions(
    const DesktopCaptureOptions& options) = default;
DesktopCaptureOptions::DesktopCaptureOptions(DesktopCaptureOptions&& options) =
    default;
DesktopCaptureOptions::~DesktopCaptureOptions() {}

DesktopCaptureOptions& DesktopCaptureOptions::operator=(
    const DesktopCaptureOptions& options) = default;
DesktopCaptureOptions& DesktopCaptureOptions::operator=(
    DesktopCaptureOptions&& options) = default;

// static
DesktopCaptureOptions DesktopCaptureOptions::CreateDefault() {
  DesktopCaptureOptions result;
#if defined(WEBRTC_USE_X11)
  result.set_x_display(SharedXDisplay::CreateDefault());
#endif
#if defined(WEBRTC_USE_PIPEWIRE)
  result.set_screencast_stream(SharedScreenCastStream::CreateDefault());
#endif
#if defined(WEBRTC_MAC) && !defined(WEBRTC_IOS)
  result.set_configuration_monitor(
      rtc::make_ref_counted<DesktopConfigurationMonitor>());
  result.set_full_screen_window_detector(
      rtc::make_ref_counted<FullScreenWindowDetector>(
          CreateFullScreenMacApplicationHandler));
#elif defined(WEBRTC_WIN)
  result.set_full_screen_window_detector(
      rtc::make_ref_counted<FullScreenWindowDetector>(
          CreateFullScreenWinApplicationHandler));
#endif
  return result;
}

}  // namespace webrtc
