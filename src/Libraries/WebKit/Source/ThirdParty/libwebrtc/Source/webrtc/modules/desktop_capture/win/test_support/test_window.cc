/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "modules/desktop_capture/win/test_support/test_window.h"

namespace webrtc {
namespace {

const WCHAR kWindowClass[] = L"DesktopCaptureTestWindowClass";
const int kWindowHeight = 200;
const int kWindowWidth = 300;

LRESULT CALLBACK WindowProc(HWND hwnd,
                            UINT msg,
                            WPARAM w_param,
                            LPARAM l_param) {
  switch (msg) {
    case WM_PAINT:
      PAINTSTRUCT paint_struct;
      HDC hdc = BeginPaint(hwnd, &paint_struct);

      // Paint the window so the color is consistent and we can inspect the
      // pixels in tests and know what to expect.
      FillRect(hdc, &paint_struct.rcPaint,
               CreateSolidBrush(RGB(kTestWindowRValue, kTestWindowGValue,
                                    kTestWindowBValue)));

      EndPaint(hwnd, &paint_struct);
  }
  return DefWindowProc(hwnd, msg, w_param, l_param);
}

}  // namespace

WindowInfo CreateTestWindow(const WCHAR* window_title,
                            const int height,
                            const int width,
                            const LONG extended_styles) {
  WindowInfo info;
  ::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&WindowProc),
                       &info.window_instance);

  WNDCLASSEXW wcex;
  memset(&wcex, 0, sizeof(wcex));
  wcex.cbSize = sizeof(wcex);
  wcex.style = CS_HREDRAW | CS_VREDRAW;
  wcex.hInstance = info.window_instance;
  wcex.lpfnWndProc = &WindowProc;
  wcex.lpszClassName = kWindowClass;
  info.window_class = ::RegisterClassExW(&wcex);

  // Use the default height and width if the caller did not supply the optional
  // height and width parameters, or if they supplied invalid values.
  int window_height = height <= 0 ? kWindowHeight : height;
  int window_width = width <= 0 ? kWindowWidth : width;
  info.hwnd =
      ::CreateWindowExW(extended_styles, kWindowClass, window_title,
                        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                        window_width, window_height, /*parent_window=*/nullptr,
                        /*menu_bar=*/nullptr, info.window_instance,
                        /*additional_params=*/nullptr);

  ::ShowWindow(info.hwnd, SW_SHOWNORMAL);
  ::UpdateWindow(info.hwnd);
  return info;
}

void ResizeTestWindow(const HWND hwnd, const int width, const int height) {
  // SWP_NOMOVE results in the x and y params being ignored.
  ::SetWindowPos(hwnd, HWND_TOP, /*x-coord=*/0, /*y-coord=*/0, width, height,
                 SWP_SHOWWINDOW | SWP_NOMOVE);
  ::UpdateWindow(hwnd);
}

void MoveTestWindow(const HWND hwnd, const int x, const int y) {
  // SWP_NOSIZE results in the width and height params being ignored.
  ::SetWindowPos(hwnd, HWND_TOP, x, y, /*width=*/0, /*height=*/0,
                 SWP_SHOWWINDOW | SWP_NOSIZE);
  ::UpdateWindow(hwnd);
}

void MinimizeTestWindow(const HWND hwnd) {
  ::ShowWindow(hwnd, SW_MINIMIZE);
}

void UnminimizeTestWindow(const HWND hwnd) {
  ::OpenIcon(hwnd);
}

void DestroyTestWindow(WindowInfo info) {
  ::DestroyWindow(info.hwnd);
  ::UnregisterClass(MAKEINTATOM(info.window_class), info.window_instance);
}

}  // namespace webrtc
