/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
#include "modules/desktop_capture/test_utils.h"

#include <stdint.h>

#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/rgba_color.h"
#include "rtc_base/checks.h"
#include "test/gtest.h"

namespace webrtc {

namespace {

void PaintDesktopFrame(DesktopFrame* frame,
                       DesktopVector pos,
                       RgbaColor color) {
  ASSERT_TRUE(frame);
  ASSERT_TRUE(DesktopRect::MakeSize(frame->size()).Contains(pos));
  *reinterpret_cast<uint32_t*>(frame->GetFrameDataAtPos(pos)) =
      color.ToUInt32();
}

// A DesktopFrame implementation to store data in heap, but the stide is
// doubled.
class DoubleSizeDesktopFrame : public DesktopFrame {
 public:
  explicit DoubleSizeDesktopFrame(DesktopSize size);
  ~DoubleSizeDesktopFrame() override;
};

DoubleSizeDesktopFrame::DoubleSizeDesktopFrame(DesktopSize size)
    : DesktopFrame(
          size,
          kBytesPerPixel * size.width() * 2,
          new uint8_t[kBytesPerPixel * size.width() * size.height() * 2],
          nullptr) {}

DoubleSizeDesktopFrame::~DoubleSizeDesktopFrame() {
  delete[] data_;
}

}  // namespace

TEST(TestUtilsTest, BasicDataEqualsCases) {
  BasicDesktopFrame frame(DesktopSize(4, 4));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      PaintDesktopFrame(&frame, DesktopVector(i, j), RgbaColor(4U * j + i));
    }
  }

  ASSERT_TRUE(DesktopFrameDataEquals(frame, frame));
  BasicDesktopFrame other(DesktopSize(4, 4));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      PaintDesktopFrame(&other, DesktopVector(i, j), RgbaColor(4U * j + i));
    }
  }
  ASSERT_TRUE(DesktopFrameDataEquals(frame, other));
  PaintDesktopFrame(&other, DesktopVector(2, 2), RgbaColor(0U));
  ASSERT_FALSE(DesktopFrameDataEquals(frame, other));
}

TEST(TestUtilsTest, DifferentSizeShouldNotEqual) {
  BasicDesktopFrame frame(DesktopSize(4, 4));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      PaintDesktopFrame(&frame, DesktopVector(i, j), RgbaColor(4U * j + i));
    }
  }

  BasicDesktopFrame other(DesktopSize(2, 8));
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 8; j++) {
      PaintDesktopFrame(&other, DesktopVector(i, j), RgbaColor(2U * j + i));
    }
  }

  ASSERT_FALSE(DesktopFrameDataEquals(frame, other));
}

TEST(TestUtilsTest, DifferentStrideShouldBeComparable) {
  BasicDesktopFrame frame(DesktopSize(4, 4));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      PaintDesktopFrame(&frame, DesktopVector(i, j), RgbaColor(4U * j + i));
    }
  }

  ASSERT_TRUE(DesktopFrameDataEquals(frame, frame));
  DoubleSizeDesktopFrame other(DesktopSize(4, 4));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      PaintDesktopFrame(&other, DesktopVector(i, j), RgbaColor(4U * j + i));
    }
  }
  ASSERT_TRUE(DesktopFrameDataEquals(frame, other));
}

}  // namespace webrtc
