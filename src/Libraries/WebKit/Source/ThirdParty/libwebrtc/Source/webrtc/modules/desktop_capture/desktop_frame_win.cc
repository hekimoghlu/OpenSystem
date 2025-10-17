/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#include "modules/desktop_capture/desktop_frame_win.h"

#include <utility>

#include "rtc_base/logging.h"

namespace webrtc {

DesktopFrameWin::DesktopFrameWin(DesktopSize size,
                                 int stride,
                                 uint8_t* data,
                                 std::unique_ptr<SharedMemory> shared_memory,
                                 HBITMAP bitmap)
    : DesktopFrame(size, stride, data, shared_memory.get()),
      bitmap_(bitmap),
      owned_shared_memory_(std::move(shared_memory)) {}

DesktopFrameWin::~DesktopFrameWin() {
  DeleteObject(bitmap_);
}

// static
std::unique_ptr<DesktopFrameWin> DesktopFrameWin::Create(
    DesktopSize size,
    SharedMemoryFactory* shared_memory_factory,
    HDC hdc) {
  int bytes_per_row = size.width() * kBytesPerPixel;
  int buffer_size = bytes_per_row * size.height();

  // Describe a device independent bitmap (DIB) that is the size of the desktop.
  BITMAPINFO bmi = {};
  bmi.bmiHeader.biHeight = -size.height();
  bmi.bmiHeader.biWidth = size.width();
  bmi.bmiHeader.biPlanes = 1;
  bmi.bmiHeader.biBitCount = DesktopFrameWin::kBytesPerPixel * 8;
  bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
  bmi.bmiHeader.biSizeImage = bytes_per_row * size.height();

  std::unique_ptr<SharedMemory> shared_memory;
  HANDLE section_handle = nullptr;
  if (shared_memory_factory) {
    shared_memory = shared_memory_factory->CreateSharedMemory(buffer_size);
    if (!shared_memory) {
      RTC_LOG(LS_WARNING) << "Failed to allocate shared memory";
      return nullptr;
    }
    section_handle = shared_memory->handle();
  }
  void* data = nullptr;
  HBITMAP bitmap =
      CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &data, section_handle, 0);
  if (!bitmap) {
    RTC_LOG(LS_WARNING) << "Failed to allocate new window frame "
                        << GetLastError();
    return nullptr;
  }

  return std::unique_ptr<DesktopFrameWin>(
      new DesktopFrameWin(size, bytes_per_row, reinterpret_cast<uint8_t*>(data),
                          std::move(shared_memory), bitmap));
}

}  // namespace webrtc
