/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_X11_MOUSE_CURSOR_MONITOR_X11_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_X11_MOUSE_CURSOR_MONITOR_X11_H_

#include <X11/X.h>

#include <memory>

#include "api/scoped_refptr.h"
#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/linux/x11/shared_x_display.h"
#include "modules/desktop_capture/mouse_cursor.h"
#include "modules/desktop_capture/mouse_cursor_monitor.h"

namespace webrtc {

class MouseCursorMonitorX11 : public MouseCursorMonitor,
                              public SharedXDisplay::XEventHandler {
 public:
  MouseCursorMonitorX11(const DesktopCaptureOptions& options, Window window);
  ~MouseCursorMonitorX11() override;

  static MouseCursorMonitor* CreateForWindow(
      const DesktopCaptureOptions& options,
      WindowId window);
  static MouseCursorMonitor* CreateForScreen(
      const DesktopCaptureOptions& options,
      ScreenId screen);
  static std::unique_ptr<MouseCursorMonitor> Create(
      const DesktopCaptureOptions& options);

  void Init(Callback* callback, Mode mode) override;
  void Capture() override;

 private:
  // SharedXDisplay::XEventHandler interface.
  bool HandleXEvent(const XEvent& event) override;

  Display* display() { return x_display_->display(); }

  // Captures current cursor shape and stores it in `cursor_shape_`.
  void CaptureCursor();

  rtc::scoped_refptr<SharedXDisplay> x_display_;
  Callback* callback_;
  Mode mode_;
  Window window_;

  bool have_xfixes_;
  int xfixes_event_base_;
  int xfixes_error_base_;

  std::unique_ptr<MouseCursor> cursor_shape_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_X11_MOUSE_CURSOR_MONITOR_X11_H_
