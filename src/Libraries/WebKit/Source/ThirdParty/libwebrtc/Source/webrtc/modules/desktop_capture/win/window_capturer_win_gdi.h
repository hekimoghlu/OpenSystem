/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_WINDOW_CAPTURER_WIN_GDI_H_
#define MODULES_DESKTOP_CAPTURE_WIN_WINDOW_CAPTURER_WIN_GDI_H_

#include <map>
#include <memory>
#include <vector>

#include "modules/desktop_capture/desktop_capture_options.h"
#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/win/window_capture_utils.h"
#include "modules/desktop_capture/window_finder_win.h"

namespace webrtc {

class WindowCapturerWinGdi : public DesktopCapturer {
 public:
  explicit WindowCapturerWinGdi(bool enumerate_current_process_windows);

  // Disallow copy and assign
  WindowCapturerWinGdi(const WindowCapturerWinGdi&) = delete;
  WindowCapturerWinGdi& operator=(const WindowCapturerWinGdi&) = delete;

  ~WindowCapturerWinGdi() override;

  static std::unique_ptr<DesktopCapturer> CreateRawWindowCapturer(
      const DesktopCaptureOptions& options);

  // DesktopCapturer interface.
  void Start(Callback* callback) override;
  void CaptureFrame() override;
  bool GetSourceList(SourceList* sources) override;
  bool SelectSource(SourceId id) override;
  bool FocusOnSelectedSource() override;
  bool IsOccluded(const DesktopVector& pos) override;

 private:
  struct CaptureResults {
    Result result;
    std::unique_ptr<DesktopFrame> frame;
  };

  CaptureResults CaptureFrame(bool capture_owned_windows);

  Callback* callback_ = nullptr;

  // HWND and HDC for the currently selected window or nullptr if window is not
  // selected.
  HWND window_ = nullptr;

  DesktopSize previous_size_;

  WindowCaptureHelperWin window_capture_helper_;

  bool enumerate_current_process_windows_;

  // This map is used to avoid flickering for the case when SelectWindow() calls
  // are interleaved with Capture() calls.
  std::map<HWND, DesktopSize> window_size_map_;

  WindowFinderWin window_finder_;

  std::vector<HWND> owned_windows_;
  std::unique_ptr<WindowCapturerWinGdi> owned_window_capturer_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_WINDOW_CAPTURER_WIN_GDI_H_
