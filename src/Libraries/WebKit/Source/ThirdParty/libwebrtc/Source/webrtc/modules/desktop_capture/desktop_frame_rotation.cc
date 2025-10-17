/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#include "modules/desktop_capture/desktop_frame_rotation.h"

#include "rtc_base/checks.h"
#include "third_party/libyuv/include/libyuv/rotate_argb.h"

namespace webrtc {

namespace {

libyuv::RotationMode ToLibyuvRotationMode(Rotation rotation) {
  switch (rotation) {
    case Rotation::CLOCK_WISE_0:
      return libyuv::kRotate0;
    case Rotation::CLOCK_WISE_90:
      return libyuv::kRotate90;
    case Rotation::CLOCK_WISE_180:
      return libyuv::kRotate180;
    case Rotation::CLOCK_WISE_270:
      return libyuv::kRotate270;
  }
  RTC_DCHECK_NOTREACHED();
  return libyuv::kRotate0;
}

DesktopRect RotateAndOffsetRect(DesktopRect rect,
                                DesktopSize size,
                                Rotation rotation,
                                DesktopVector offset) {
  DesktopRect result = RotateRect(rect, size, rotation);
  result.Translate(offset);
  return result;
}

}  // namespace

Rotation ReverseRotation(Rotation rotation) {
  switch (rotation) {
    case Rotation::CLOCK_WISE_0:
      return rotation;
    case Rotation::CLOCK_WISE_90:
      return Rotation::CLOCK_WISE_270;
    case Rotation::CLOCK_WISE_180:
      return Rotation::CLOCK_WISE_180;
    case Rotation::CLOCK_WISE_270:
      return Rotation::CLOCK_WISE_90;
  }
  RTC_DCHECK_NOTREACHED();
  return Rotation::CLOCK_WISE_0;
}

DesktopSize RotateSize(DesktopSize size, Rotation rotation) {
  switch (rotation) {
    case Rotation::CLOCK_WISE_0:
    case Rotation::CLOCK_WISE_180:
      return size;
    case Rotation::CLOCK_WISE_90:
    case Rotation::CLOCK_WISE_270:
      return DesktopSize(size.height(), size.width());
  }
  RTC_DCHECK_NOTREACHED();
  return DesktopSize();
}

DesktopRect RotateRect(DesktopRect rect, DesktopSize size, Rotation rotation) {
  switch (rotation) {
    case Rotation::CLOCK_WISE_0:
      return rect;
    case Rotation::CLOCK_WISE_90:
      return DesktopRect::MakeXYWH(size.height() - rect.bottom(), rect.left(),
                                   rect.height(), rect.width());
    case Rotation::CLOCK_WISE_180:
      return DesktopRect::MakeXYWH(size.width() - rect.right(),
                                   size.height() - rect.bottom(), rect.width(),
                                   rect.height());
    case Rotation::CLOCK_WISE_270:
      return DesktopRect::MakeXYWH(rect.top(), size.width() - rect.right(),
                                   rect.height(), rect.width());
  }
  RTC_DCHECK_NOTREACHED();
  return DesktopRect();
}

void RotateDesktopFrame(const DesktopFrame& source,
                        const DesktopRect& source_rect,
                        const Rotation& rotation,
                        const DesktopVector& target_offset,
                        DesktopFrame* target) {
  RTC_DCHECK(target);
  RTC_DCHECK(DesktopRect::MakeSize(source.size()).ContainsRect(source_rect));
  // The rectangle in `target`.
  const DesktopRect target_rect =
      RotateAndOffsetRect(source_rect, source.size(), rotation, target_offset);
  RTC_DCHECK(DesktopRect::MakeSize(target->size()).ContainsRect(target_rect));

  if (target_rect.is_empty()) {
    return;
  }

  int result = libyuv::ARGBRotate(
      source.GetFrameDataAtPos(source_rect.top_left()), source.stride(),
      target->GetFrameDataAtPos(target_rect.top_left()), target->stride(),
      source_rect.width(), source_rect.height(),
      ToLibyuvRotationMode(rotation));
  RTC_DCHECK_EQ(result, 0);
}

}  // namespace webrtc
