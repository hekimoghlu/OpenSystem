/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#include <memory>

#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/mouse_cursor_monitor.h"

#if defined(WEBRTC_USE_X11)
#include "modules/desktop_capture/linux/x11/mouse_cursor_monitor_x11.h"
#endif  // defined(WEBRTC_USE_X11)

#if defined(WEBRTC_USE_PIPEWIRE)
#include "modules/desktop_capture/linux/wayland/mouse_cursor_monitor_pipewire.h"
#endif  // defined(WEBRTC_USE_PIPEWIRE)

namespace webrtc {

// static
MouseCursorMonitor* MouseCursorMonitor::CreateForWindow(
    const DesktopCaptureOptions& options,
    WindowId window) {
#if defined(WEBRTC_USE_X11)
  return MouseCursorMonitorX11::CreateForWindow(options, window);
#else
  return nullptr;
#endif  // defined(WEBRTC_USE_X11)
}

// static
MouseCursorMonitor* MouseCursorMonitor::CreateForScreen(
    const DesktopCaptureOptions& options,
    ScreenId screen) {
#if defined(WEBRTC_USE_X11)
  return MouseCursorMonitorX11::CreateForScreen(options, screen);
#else
  return nullptr;
#endif  // defined(WEBRTC_USE_X11)
}

// static
std::unique_ptr<MouseCursorMonitor> MouseCursorMonitor::Create(
    const DesktopCaptureOptions& options) {
#if defined(WEBRTC_USE_PIPEWIRE)
  if (options.allow_pipewire() && DesktopCapturer::IsRunningUnderWayland() &&
      options.screencast_stream()) {
    return std::make_unique<MouseCursorMonitorPipeWire>(options);
  }
#endif  // defined(WEBRTC_USE_PIPEWIRE)

#if defined(WEBRTC_USE_X11)
  return MouseCursorMonitorX11::Create(options);
#else
  return nullptr;
#endif  // defined(WEBRTC_USE_X11)
}

}  // namespace webrtc
