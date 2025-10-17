/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_LINUX_X11_WINDOW_CAPTURER_X11_H_
#define MODULES_DESKTOP_CAPTURE_LINUX_X11_WINDOW_CAPTURER_X11_H_

#include <X11/X.h>
#include <X11/Xlib.h>

#include <memory>
#include <string>

#include "api/scoped_refptr.h"
#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/linux/x11/shared_x_display.h"
#include "modules/desktop_capture/linux/x11/window_finder_x11.h"
#include "modules/desktop_capture/linux/x11/x_atom_cache.h"
#include "modules/desktop_capture/linux/x11/x_server_pixel_buffer.h"

namespace webrtc {

class WindowCapturerX11 : public DesktopCapturer,
                          public SharedXDisplay::XEventHandler {
 public:
  explicit WindowCapturerX11(const DesktopCaptureOptions& options);
  ~WindowCapturerX11() override;

  WindowCapturerX11(const WindowCapturerX11&) = delete;
  WindowCapturerX11& operator=(const WindowCapturerX11&) = delete;

  static std::unique_ptr<DesktopCapturer> CreateRawWindowCapturer(
      const DesktopCaptureOptions& options);

  // DesktopCapturer interface.
  void Start(Callback* callback) override;
  void CaptureFrame() override;
  bool GetSourceList(SourceList* sources) override;
  bool SelectSource(SourceId id) override;
  bool FocusOnSelectedSource() override;
  bool IsOccluded(const DesktopVector& pos) override;

  // SharedXDisplay::XEventHandler interface.
  bool HandleXEvent(const XEvent& event) override;

 private:
  Display* display() { return x_display_->display(); }

  // Returns window title for the specified X `window`.
  bool GetWindowTitle(::Window window, std::string* title);

  Callback* callback_ = nullptr;

  rtc::scoped_refptr<SharedXDisplay> x_display_;

  bool has_composite_extension_ = false;

  ::Window selected_window_ = 0;
  XServerPixelBuffer x_server_pixel_buffer_;
  XAtomCache atom_cache_;
  WindowFinderX11 window_finder_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_LINUX_X11_WINDOW_CAPTURER_X11_H_
