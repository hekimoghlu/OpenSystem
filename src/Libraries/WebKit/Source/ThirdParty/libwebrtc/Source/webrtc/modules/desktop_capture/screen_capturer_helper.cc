/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "modules/desktop_capture/screen_capturer_helper.h"

namespace webrtc {

void ScreenCapturerHelper::ClearInvalidRegion() {
  MutexLock scoped_invalid_region_lock(&invalid_region_mutex_);
  invalid_region_.Clear();
}

void ScreenCapturerHelper::InvalidateRegion(
    const DesktopRegion& invalid_region) {
  MutexLock scoped_invalid_region_lock(&invalid_region_mutex_);
  invalid_region_.AddRegion(invalid_region);
}

void ScreenCapturerHelper::InvalidateScreen(const DesktopSize& size) {
  MutexLock scoped_invalid_region_lock(&invalid_region_mutex_);
  invalid_region_.AddRect(DesktopRect::MakeSize(size));
}

void ScreenCapturerHelper::TakeInvalidRegion(DesktopRegion* invalid_region) {
  invalid_region->Clear();

  {
    MutexLock scoped_invalid_region_lock(&invalid_region_mutex_);
    invalid_region->Swap(&invalid_region_);
  }

  if (log_grid_size_ > 0) {
    DesktopRegion expanded_region;
    ExpandToGrid(*invalid_region, log_grid_size_, &expanded_region);
    expanded_region.Swap(invalid_region);

    invalid_region->IntersectWith(DesktopRect::MakeSize(size_most_recent_));
  }
}

void ScreenCapturerHelper::SetLogGridSize(int log_grid_size) {
  log_grid_size_ = log_grid_size;
}

const DesktopSize& ScreenCapturerHelper::size_most_recent() const {
  return size_most_recent_;
}

void ScreenCapturerHelper::set_size_most_recent(const DesktopSize& size) {
  size_most_recent_ = size;
}

// Returns the largest multiple of `n` that is <= `x`.
// `n` must be a power of 2. `nMask` is ~(`n` - 1).
static int DownToMultiple(int x, int nMask) {
  return (x & nMask);
}

// Returns the smallest multiple of `n` that is >= `x`.
// `n` must be a power of 2. `nMask` is ~(`n` - 1).
static int UpToMultiple(int x, int n, int nMask) {
  return ((x + n - 1) & nMask);
}

void ScreenCapturerHelper::ExpandToGrid(const DesktopRegion& region,
                                        int log_grid_size,
                                        DesktopRegion* result) {
  RTC_DCHECK_GE(log_grid_size, 1);
  int grid_size = 1 << log_grid_size;
  int grid_size_mask = ~(grid_size - 1);

  result->Clear();
  for (DesktopRegion::Iterator it(region); !it.IsAtEnd(); it.Advance()) {
    int left = DownToMultiple(it.rect().left(), grid_size_mask);
    int right = UpToMultiple(it.rect().right(), grid_size, grid_size_mask);
    int top = DownToMultiple(it.rect().top(), grid_size_mask);
    int bottom = UpToMultiple(it.rect().bottom(), grid_size, grid_size_mask);
    result->AddRect(DesktopRect::MakeLTRB(left, top, right, bottom));
  }
}

}  // namespace webrtc
