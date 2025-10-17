/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURER_WIN_GDI_H_
#define MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURER_WIN_GDI_H_

#include <windows.h>

#include <memory>

#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/screen_capture_frame_queue.h"
#include "modules/desktop_capture/shared_desktop_frame.h"
#include "modules/desktop_capture/win/display_configuration_monitor.h"
#include "modules/desktop_capture/win/scoped_thread_desktop.h"

namespace webrtc {

// ScreenCapturerWinGdi captures 32bit RGB using GDI.
//
// ScreenCapturerWinGdi is double-buffered as required by ScreenCapturer.
// This class does not detect DesktopFrame::updated_region(), the field is
// always set to the entire frame rectangle. ScreenCapturerDifferWrapper should
// be used if that functionality is necessary.
class ScreenCapturerWinGdi : public DesktopCapturer {
 public:
  explicit ScreenCapturerWinGdi(const DesktopCaptureOptions& options);
  ~ScreenCapturerWinGdi() override;

  ScreenCapturerWinGdi(const ScreenCapturerWinGdi&) = delete;
  ScreenCapturerWinGdi& operator=(const ScreenCapturerWinGdi&) = delete;

  // Overridden from ScreenCapturer:
  void Start(Callback* callback) override;
  void SetSharedMemoryFactory(
      std::unique_ptr<SharedMemoryFactory> shared_memory_factory) override;
  void CaptureFrame() override;
  bool GetSourceList(SourceList* sources) override;
  bool SelectSource(SourceId id) override;

 private:
  typedef HRESULT(WINAPI* DwmEnableCompositionFunc)(UINT);

  // Make sure that the device contexts match the screen configuration.
  void PrepareCaptureResources();

  // Captures the current screen contents into the current buffer. Returns true
  // if succeeded.
  bool CaptureImage();

  // Capture the current cursor shape.
  void CaptureCursor();

  Callback* callback_ = nullptr;
  std::unique_ptr<SharedMemoryFactory> shared_memory_factory_;
  SourceId current_screen_id_ = kFullDesktopScreenId;
  std::wstring current_device_key_;

  ScopedThreadDesktop desktop_;

  // GDI resources used for screen capture.
  HDC desktop_dc_ = NULL;
  HDC memory_dc_ = NULL;

  // Queue of the frames buffers.
  ScreenCaptureFrameQueue<SharedDesktopFrame> queue_;

  DisplayConfigurationMonitor display_configuration_monitor_;

  HMODULE dwmapi_library_ = NULL;
  DwmEnableCompositionFunc composition_func_ = nullptr;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURER_WIN_GDI_H_
