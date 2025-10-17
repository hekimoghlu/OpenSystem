/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include "modules/desktop_capture/mac/desktop_frame_provider.h"

#include <utility>

#include "modules/desktop_capture/mac/desktop_frame_cgimage.h"
#include "modules/desktop_capture/mac/desktop_frame_iosurface.h"

namespace webrtc {

DesktopFrameProvider::DesktopFrameProvider(bool allow_iosurface)
    : allow_iosurface_(allow_iosurface) {
  thread_checker_.Detach();
}

DesktopFrameProvider::~DesktopFrameProvider() {
  RTC_DCHECK(thread_checker_.IsCurrent());

  Release();
}

std::unique_ptr<DesktopFrame> DesktopFrameProvider::TakeLatestFrameForDisplay(
    CGDirectDisplayID display_id) {
  RTC_DCHECK(thread_checker_.IsCurrent());

  if (!allow_iosurface_ || !io_surfaces_[display_id]) {
    // Regenerate a snapshot. If iosurface is on it will be empty until the
    // stream handler is called.
    return DesktopFrameCGImage::CreateForDisplay(display_id);
  }

  return io_surfaces_[display_id]->Share();
}

void DesktopFrameProvider::InvalidateIOSurface(CGDirectDisplayID display_id,
                                               rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface) {
  RTC_DCHECK(thread_checker_.IsCurrent());

  if (!allow_iosurface_) {
    return;
  }

  std::unique_ptr<DesktopFrameIOSurface> desktop_frame_iosurface =
      DesktopFrameIOSurface::Wrap(io_surface);

  io_surfaces_[display_id] = desktop_frame_iosurface ?
      SharedDesktopFrame::Wrap(std::move(desktop_frame_iosurface)) :
      nullptr;
}

void DesktopFrameProvider::Release() {
  RTC_DCHECK(thread_checker_.IsCurrent());

  if (!allow_iosurface_) {
    return;
  }

  io_surfaces_.clear();
}

}  // namespace webrtc
