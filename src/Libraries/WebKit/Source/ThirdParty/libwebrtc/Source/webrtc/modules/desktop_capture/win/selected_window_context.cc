/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include "modules/desktop_capture/win/selected_window_context.h"

namespace webrtc {

SelectedWindowContext::SelectedWindowContext(
    HWND selected_window,
    DesktopRect selected_window_rect,
    WindowCaptureHelperWin* window_capture_helper)
    : selected_window_(selected_window),
      selected_window_rect_(selected_window_rect),
      window_capture_helper_(window_capture_helper) {
  selected_window_thread_id_ =
      GetWindowThreadProcessId(selected_window, &selected_window_process_id_);
}

bool SelectedWindowContext::IsSelectedWindowValid() const {
  return selected_window_thread_id_ != 0;
}

bool SelectedWindowContext::IsWindowOwnedBySelectedWindow(HWND hwnd) const {
  // This check works for drop-down menus & dialog pop-up windows.
  if (GetAncestor(hwnd, GA_ROOTOWNER) == selected_window_) {
    return true;
  }

  // Assume that all other windows are unrelated to the selected window.
  // This will cause some windows that are actually related to be missed,
  // e.g. context menus and tool-tips, but avoids the risk of capturing
  // unrelated windows. Using heuristics such as matching the thread and
  // process Ids suffers from false-positives, e.g. in multi-document
  // applications.

  return false;
}

bool SelectedWindowContext::IsWindowOverlappingSelectedWindow(HWND hwnd) const {
  return window_capture_helper_->AreWindowsOverlapping(hwnd, selected_window_,
                                                       selected_window_rect_);
}

HWND SelectedWindowContext::selected_window() const {
  return selected_window_;
}

WindowCaptureHelperWin* SelectedWindowContext::window_capture_helper() const {
  return window_capture_helper_;
}

}  // namespace webrtc
