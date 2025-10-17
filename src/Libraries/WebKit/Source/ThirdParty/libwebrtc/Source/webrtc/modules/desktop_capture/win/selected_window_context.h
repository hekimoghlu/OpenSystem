/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_SELECTED_WINDOW_CONTEXT_H_
#define MODULES_DESKTOP_CAPTURE_WIN_SELECTED_WINDOW_CONTEXT_H_

#include <windows.h>

#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/win/window_capture_utils.h"

namespace webrtc {

class SelectedWindowContext {
 public:
  SelectedWindowContext(HWND selected_window,
                        DesktopRect selected_window_rect,
                        WindowCaptureHelperWin* window_capture_helper);

  bool IsSelectedWindowValid() const;

  bool IsWindowOwnedBySelectedWindow(HWND hwnd) const;
  bool IsWindowOverlappingSelectedWindow(HWND hwnd) const;

  HWND selected_window() const;
  WindowCaptureHelperWin* window_capture_helper() const;

 private:
  const HWND selected_window_;
  const DesktopRect selected_window_rect_;
  WindowCaptureHelperWin* const window_capture_helper_;
  DWORD selected_window_thread_id_;
  DWORD selected_window_process_id_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_SELECTED_WINDOW_CONTEXT_H_
