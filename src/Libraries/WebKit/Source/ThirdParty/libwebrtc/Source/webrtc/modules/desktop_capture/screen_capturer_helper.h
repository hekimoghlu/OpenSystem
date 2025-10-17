/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#ifndef MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURER_HELPER_H_
#define MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURER_HELPER_H_

#include <memory>

#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/desktop_region.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

// ScreenCapturerHelper is intended to be used by an implementation of the
// ScreenCapturer interface. It maintains a thread-safe invalid region, and
// the size of the most recently captured screen, on behalf of the
// ScreenCapturer that owns it.
class ScreenCapturerHelper {
 public:
  ScreenCapturerHelper() = default;
  ~ScreenCapturerHelper() = default;

  ScreenCapturerHelper(const ScreenCapturerHelper&) = delete;
  ScreenCapturerHelper& operator=(const ScreenCapturerHelper&) = delete;

  // Clear out the invalid region.
  void ClearInvalidRegion();

  // Invalidate the specified region.
  void InvalidateRegion(const DesktopRegion& invalid_region);

  // Invalidate the entire screen, of a given size.
  void InvalidateScreen(const DesktopSize& size);

  // Copies current invalid region to `invalid_region` clears invalid region
  // storage for the next frame.
  void TakeInvalidRegion(DesktopRegion* invalid_region);

  // Access the size of the most recently captured screen.
  const DesktopSize& size_most_recent() const;
  void set_size_most_recent(const DesktopSize& size);

  // Lossy compression can result in color values leaking between pixels in one
  // block. If part of a block changes, then unchanged parts of that block can
  // be changed in the compressed output. So we need to re-render an entire
  // block whenever part of the block changes.
  //
  // If `log_grid_size` is >= 1, then this function makes TakeInvalidRegion()
  // produce an invalid region expanded so that its vertices lie on a grid of
  // size 2 ^ `log_grid_size`. The expanded region is then clipped to the size
  // of the most recently captured screen, as previously set by
  // set_size_most_recent().
  // If `log_grid_size` is <= 0, then the invalid region is not expanded.
  void SetLogGridSize(int log_grid_size);

  // Expands a region so that its vertices all lie on a grid.
  // The grid size must be >= 2, so `log_grid_size` must be >= 1.
  static void ExpandToGrid(const DesktopRegion& region,
                           int log_grid_size,
                           DesktopRegion* result);

 private:
  // A region that has been manually invalidated (through InvalidateRegion).
  // These will be returned as dirty_region in the capture data during the next
  // capture.
  DesktopRegion invalid_region_ RTC_GUARDED_BY(invalid_region_mutex_);

  // A lock protecting `invalid_region_` across threads.
  Mutex invalid_region_mutex_;

  // The size of the most recently captured screen.
  DesktopSize size_most_recent_;

  // The log (base 2) of the size of the grid to which the invalid region is
  // expanded.
  // If the value is <= 0, then the invalid region is not expanded to a grid.
  int log_grid_size_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_SCREEN_CAPTURER_HELPER_H_
