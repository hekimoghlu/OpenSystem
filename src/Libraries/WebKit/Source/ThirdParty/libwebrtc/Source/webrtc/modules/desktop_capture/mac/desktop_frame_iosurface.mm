/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#include "modules/desktop_capture/mac/desktop_frame_iosurface.h"

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

// static
std::unique_ptr<DesktopFrameIOSurface> DesktopFrameIOSurface::Wrap(
    rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface) {
  if (!io_surface) {
    return nullptr;
  }

  IOSurfaceIncrementUseCount(io_surface.get());
  IOReturn status = IOSurfaceLock(io_surface.get(), kIOSurfaceLockReadOnly, nullptr);
  if (status != kIOReturnSuccess) {
    RTC_LOG(LS_ERROR) << "Failed to lock the IOSurface with status " << status;
    IOSurfaceDecrementUseCount(io_surface.get());
    return nullptr;
  }

  // Verify that the image has 32-bit depth.
  int bytes_per_pixel = IOSurfaceGetBytesPerElement(io_surface.get());
  if (bytes_per_pixel != DesktopFrame::kBytesPerPixel) {
    RTC_LOG(LS_ERROR) << "CGDisplayStream handler returned IOSurface with " << (8 * bytes_per_pixel)
                      << " bits per pixel. Only 32-bit depth is supported.";
    IOSurfaceUnlock(io_surface.get(), kIOSurfaceLockReadOnly, nullptr);
    IOSurfaceDecrementUseCount(io_surface.get());
    return nullptr;
  }

  return std::unique_ptr<DesktopFrameIOSurface>(new DesktopFrameIOSurface(io_surface));
}

DesktopFrameIOSurface::DesktopFrameIOSurface(rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface)
    : DesktopFrame(
          DesktopSize(IOSurfaceGetWidth(io_surface.get()), IOSurfaceGetHeight(io_surface.get())),
          IOSurfaceGetBytesPerRow(io_surface.get()),
          static_cast<uint8_t*>(IOSurfaceGetBaseAddress(io_surface.get())),
          nullptr),
      io_surface_(io_surface) {
  RTC_DCHECK(io_surface_);
}

DesktopFrameIOSurface::~DesktopFrameIOSurface() {
  IOSurfaceUnlock(io_surface_.get(), kIOSurfaceLockReadOnly, nullptr);
  IOSurfaceDecrementUseCount(io_surface_.get());
}

}  // namespace webrtc
