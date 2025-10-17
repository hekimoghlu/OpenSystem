/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURE_TYPES_H_
#define MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURE_TYPES_H_

#include <stdint.h>

namespace webrtc {

enum class CaptureType { kWindow, kScreen, kAnyScreenContent };

// Type used to identify windows on the desktop. Values are platform-specific:
//   - On Windows: HWND cast to intptr_t.
//   - On Linux (with X11): X11 Window (unsigned long) type cast to intptr_t.
//   - On OSX: integer window number.
typedef intptr_t WindowId;

const WindowId kNullWindowId = 0;

const int64_t kInvalidDisplayId = -1;

// Type used to identify screens on the desktop. Values are platform-specific:
//   - On Windows: integer display device index.
//   - On OSX: CGDirectDisplayID cast to intptr_t.
//   - On Linux (with X11): TBD.
//   - On ChromeOS: display::Display::id() is an int64_t.
// On Windows, ScreenId is implementation dependent: sending a ScreenId from one
// implementation to another usually won't work correctly.
#if defined(CHROMEOS)
typedef int64_t ScreenId;
#else
typedef intptr_t ScreenId;
#endif

// The screen id corresponds to all screen combined together.
const ScreenId kFullDesktopScreenId = -1;

const ScreenId kInvalidScreenId = -2;

// Integers to attach to each DesktopFrame to differentiate the generator of
// the frame. The entries in this namespace should remain in sync with the
// SequentialDesktopCapturerId enum, which is logged via UMA.
// `kScreenCapturerWinGdi` and `kScreenCapturerWinDirectx` values are preserved
// to maintain compatibility
namespace DesktopCapturerId {
constexpr uint32_t CreateFourCC(char a, char b, char c, char d) {
  return ((static_cast<uint32_t>(a)) | (static_cast<uint32_t>(b) << 8) |
          (static_cast<uint32_t>(c) << 16) | (static_cast<uint32_t>(d) << 24));
}

constexpr uint32_t kUnknown = 0;
constexpr uint32_t kWgcCapturerWin = 1;
constexpr uint32_t kScreenCapturerWinMagnifier = 2;
constexpr uint32_t kWindowCapturerWinGdi = 3;
constexpr uint32_t kScreenCapturerWinGdi = CreateFourCC('G', 'D', 'I', ' ');
constexpr uint32_t kScreenCapturerWinDirectx = CreateFourCC('D', 'X', 'G', 'I');
constexpr uint32_t kX11CapturerLinux = CreateFourCC('X', '1', '1', ' ');
constexpr uint32_t kWaylandCapturerLinux = CreateFourCC('W', 'L', ' ', ' ');
}  // namespace DesktopCapturerId

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DESKTOP_CAPTURE_TYPES_H_
