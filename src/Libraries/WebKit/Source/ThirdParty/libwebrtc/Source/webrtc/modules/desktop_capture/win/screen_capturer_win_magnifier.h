/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURER_WIN_MAGNIFIER_H_
#define MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURER_WIN_MAGNIFIER_H_

#include <magnification.h>
#include <wincodec.h>
#include <windows.h>

#include <memory>

#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/screen_capture_frame_queue.h"
#include "modules/desktop_capture/screen_capturer_helper.h"
#include "modules/desktop_capture/shared_desktop_frame.h"
#include "modules/desktop_capture/win/scoped_thread_desktop.h"

namespace webrtc {

class DesktopFrame;
class DesktopRect;

// Captures the screen using the Magnification API to support window exclusion.
// Each capturer must run on a dedicated thread because it uses thread local
// storage for redirecting the library callback. Also the thread must have a UI
// message loop to handle the window messages for the magnifier window.
//
// This class does not detect DesktopFrame::updated_region(), the field is
// always set to the entire frame rectangle. ScreenCapturerDifferWrapper should
// be used if that functionality is necessary.
class ScreenCapturerWinMagnifier : public DesktopCapturer {
 public:
  ScreenCapturerWinMagnifier();
  ~ScreenCapturerWinMagnifier() override;

  ScreenCapturerWinMagnifier(const ScreenCapturerWinMagnifier&) = delete;
  ScreenCapturerWinMagnifier& operator=(const ScreenCapturerWinMagnifier&) =
      delete;

  // Overridden from ScreenCapturer:
  void Start(Callback* callback) override;
  void SetSharedMemoryFactory(
      std::unique_ptr<SharedMemoryFactory> shared_memory_factory) override;
  void CaptureFrame() override;
  bool GetSourceList(SourceList* screens) override;
  bool SelectSource(SourceId id) override;
  void SetExcludedWindow(WindowId window) override;

 private:
  typedef BOOL(WINAPI* MagImageScalingCallback)(HWND hwnd,
                                                void* srcdata,
                                                MAGIMAGEHEADER srcheader,
                                                void* destdata,
                                                MAGIMAGEHEADER destheader,
                                                RECT unclipped,
                                                RECT clipped,
                                                HRGN dirty);
  typedef BOOL(WINAPI* MagInitializeFunc)(void);
  typedef BOOL(WINAPI* MagUninitializeFunc)(void);
  typedef BOOL(WINAPI* MagSetWindowSourceFunc)(HWND hwnd, RECT rect);
  typedef BOOL(WINAPI* MagSetWindowFilterListFunc)(HWND hwnd,
                                                   DWORD dwFilterMode,
                                                   int count,
                                                   HWND* pHWND);
  typedef BOOL(WINAPI* MagSetImageScalingCallbackFunc)(
      HWND hwnd,
      MagImageScalingCallback callback);

  static BOOL WINAPI OnMagImageScalingCallback(HWND hwnd,
                                               void* srcdata,
                                               MAGIMAGEHEADER srcheader,
                                               void* destdata,
                                               MAGIMAGEHEADER destheader,
                                               RECT unclipped,
                                               RECT clipped,
                                               HRGN dirty);

  // Captures the screen within `rect` in the desktop coordinates. Returns true
  // if succeeded.
  // It can only capture the primary screen for now. The magnification library
  // crashes under some screen configurations (e.g. secondary screen on top of
  // primary screen) if it tries to capture a non-primary screen. The caller
  // must make sure not calling it on non-primary screens.
  bool CaptureImage(const DesktopRect& rect);

  // Helper method for setting up the magnifier control. Returns true if
  // succeeded.
  bool InitializeMagnifier();

  // Called by OnMagImageScalingCallback to output captured data.
  void OnCaptured(void* data, const MAGIMAGEHEADER& header);

  // Makes sure the current frame exists and matches `size`.
  void CreateCurrentFrameIfNecessary(const DesktopSize& size);

  Callback* callback_ = nullptr;
  std::unique_ptr<SharedMemoryFactory> shared_memory_factory_;
  ScreenId current_screen_id_ = kFullDesktopScreenId;
  std::wstring current_device_key_;
  HWND excluded_window_ = NULL;

  // Queue of the frames buffers.
  ScreenCaptureFrameQueue<SharedDesktopFrame> queue_;

  ScopedThreadDesktop desktop_;

  // Used for getting the screen dpi.
  HDC desktop_dc_ = NULL;

  HMODULE mag_lib_handle_ = NULL;
  MagInitializeFunc mag_initialize_func_ = nullptr;
  MagUninitializeFunc mag_uninitialize_func_ = nullptr;
  MagSetWindowSourceFunc set_window_source_func_ = nullptr;
  MagSetWindowFilterListFunc set_window_filter_list_func_ = nullptr;
  MagSetImageScalingCallbackFunc set_image_scaling_callback_func_ = nullptr;

  // The hidden window hosting the magnifier control.
  HWND host_window_ = NULL;
  // The magnifier control that captures the screen.
  HWND magnifier_window_ = NULL;

  // True if the magnifier control has been successfully initialized.
  bool magnifier_initialized_ = false;

  // True if the last OnMagImageScalingCallback was called and handled
  // successfully. Reset at the beginning of each CaptureImage call.
  bool magnifier_capture_succeeded_ = true;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_SCREEN_CAPTURER_WIN_MAGNIFIER_H_
