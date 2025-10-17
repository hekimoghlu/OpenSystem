/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_DESKTOP_FRAME_WIN_H_
#define MODULES_DESKTOP_CAPTURE_DESKTOP_FRAME_WIN_H_

#include <windows.h>

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"

namespace webrtc {

// DesktopFrame implementation used by screen and window captures on Windows.
// Frame data is stored in a GDI bitmap.
class DesktopFrameWin : public DesktopFrame {
 public:
  ~DesktopFrameWin() override;

  DesktopFrameWin(const DesktopFrameWin&) = delete;
  DesktopFrameWin& operator=(const DesktopFrameWin&) = delete;

  static std::unique_ptr<DesktopFrameWin>
  Create(DesktopSize size, SharedMemoryFactory* shared_memory_factory, HDC hdc);

  HBITMAP bitmap() { return bitmap_; }

 private:
  DesktopFrameWin(DesktopSize size,
                  int stride,
                  uint8_t* data,
                  std::unique_ptr<SharedMemory> shared_memory,
                  HBITMAP bitmap);

  HBITMAP bitmap_;
  std::unique_ptr<SharedMemory> owned_shared_memory_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_DESKTOP_FRAME_WIN_H_
