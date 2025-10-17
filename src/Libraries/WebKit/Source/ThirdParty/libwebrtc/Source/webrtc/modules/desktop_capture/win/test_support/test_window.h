/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_TEST_SUPPORT_TEST_WINDOW_H_
#define MODULES_DESKTOP_CAPTURE_WIN_TEST_SUPPORT_TEST_WINDOW_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace webrtc {

typedef unsigned char uint8_t;

// Define an arbitrary color for the test window with unique R, G, and B values
// so consumers can verify captured content in tests.
const uint8_t kTestWindowRValue = 191;
const uint8_t kTestWindowGValue = 99;
const uint8_t kTestWindowBValue = 12;

struct WindowInfo {
  HWND hwnd;
  HINSTANCE window_instance;
  ATOM window_class;
};

WindowInfo CreateTestWindow(const WCHAR* window_title,
                            int height = 0,
                            int width = 0,
                            LONG extended_styles = 0);

void ResizeTestWindow(HWND hwnd, int width, int height);

void MoveTestWindow(HWND hwnd, int x, int y);

void MinimizeTestWindow(HWND hwnd);

void UnminimizeTestWindow(HWND hwnd);

void DestroyTestWindow(WindowInfo info);

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_TEST_SUPPORT_TEST_WINDOW_H_
