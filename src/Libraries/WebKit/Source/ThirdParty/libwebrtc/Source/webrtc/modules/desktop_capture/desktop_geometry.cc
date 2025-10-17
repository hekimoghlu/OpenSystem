/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include "modules/desktop_capture/desktop_geometry.h"

#include <algorithm>
#include <cmath>

namespace webrtc {

bool DesktopRect::Contains(const DesktopVector& point) const {
  return point.x() >= left() && point.x() < right() && point.y() >= top() &&
         point.y() < bottom();
}

bool DesktopRect::ContainsRect(const DesktopRect& rect) const {
  return rect.left() >= left() && rect.right() <= right() &&
         rect.top() >= top() && rect.bottom() <= bottom();
}

void DesktopRect::IntersectWith(const DesktopRect& rect) {
  left_ = std::max(left(), rect.left());
  top_ = std::max(top(), rect.top());
  right_ = std::min(right(), rect.right());
  bottom_ = std::min(bottom(), rect.bottom());
  if (is_empty()) {
    left_ = 0;
    top_ = 0;
    right_ = 0;
    bottom_ = 0;
  }
}

void DesktopRect::UnionWith(const DesktopRect& rect) {
  if (is_empty()) {
    *this = rect;
    return;
  }

  if (rect.is_empty()) {
    return;
  }

  left_ = std::min(left(), rect.left());
  top_ = std::min(top(), rect.top());
  right_ = std::max(right(), rect.right());
  bottom_ = std::max(bottom(), rect.bottom());
}

void DesktopRect::Translate(int32_t dx, int32_t dy) {
  left_ += dx;
  top_ += dy;
  right_ += dx;
  bottom_ += dy;
}

void DesktopRect::Extend(int32_t left_offset,
                         int32_t top_offset,
                         int32_t right_offset,
                         int32_t bottom_offset) {
  left_ -= left_offset;
  top_ -= top_offset;
  right_ += right_offset;
  bottom_ += bottom_offset;
}

void DesktopRect::Scale(double horizontal, double vertical) {
  right_ += static_cast<int>(std::round(width() * (horizontal - 1)));
  bottom_ += static_cast<int>(std::round(height() * (vertical - 1)));
}

}  // namespace webrtc
