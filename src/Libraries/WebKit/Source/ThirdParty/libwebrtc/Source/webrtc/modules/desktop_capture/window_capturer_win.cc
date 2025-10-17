/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/win/window_capturer_win_gdi.h"

#if defined(RTC_ENABLE_WIN_WGC)
#include "modules/desktop_capture/blank_detector_desktop_capturer_wrapper.h"
#include "modules/desktop_capture/fallback_desktop_capturer_wrapper.h"
#include "modules/desktop_capture/win/wgc_capturer_win.h"
#include "rtc_base/win/windows_version.h"
#endif  // defined(RTC_ENABLE_WIN_WGC)

namespace webrtc {

// static
std::unique_ptr<DesktopCapturer> DesktopCapturer::CreateRawWindowCapturer(
    const DesktopCaptureOptions& options) {
  std::unique_ptr<DesktopCapturer> capturer(
      WindowCapturerWinGdi::CreateRawWindowCapturer(options));
#if defined(RTC_ENABLE_WIN_WGC)
  if (options.allow_wgc_capturer_fallback() &&
      rtc::rtc_win::GetVersion() >= rtc::rtc_win::Version::VERSION_WIN11) {
    // BlankDectector capturer will send an error when it detects a failed
    // GDI rendering, then Fallback capturer will try to capture it again with
    // WGC.
    capturer = std::make_unique<BlankDetectorDesktopCapturerWrapper>(
        std::move(capturer), RgbaColor(0, 0, 0, 0),
        /*check_per_capture*/ true);

    capturer = std::make_unique<FallbackDesktopCapturerWrapper>(
        std::move(capturer),
        WgcCapturerWin::CreateRawWindowCapturer(
            options, /*allow_delayed_capturable_check*/ true));
  }
#endif  // defined(RTC_ENABLE_WIN_WGC)
  return capturer;
}

}  // namespace webrtc
