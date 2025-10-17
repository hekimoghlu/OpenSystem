/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
#include "modules/desktop_capture/win/cursor.h"

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/mouse_cursor.h"
#include "modules/desktop_capture/win/cursor_unittest_resources.h"
#include "modules/desktop_capture/win/scoped_gdi_object.h"
#include "test/gmock.h"

namespace webrtc {

namespace {

// Loads `left` from resources, converts it to a `MouseCursor` instance and
// compares pixels with `right`. Returns true of MouseCursor bits match `right`.
// `right` must be a 32bpp cursor with alpha channel.
bool ConvertToMouseShapeAndCompare(unsigned left, unsigned right) {
  HMODULE instance = GetModuleHandle(NULL);

  // Load `left` from the EXE module's resources.
  win::ScopedCursor cursor(reinterpret_cast<HCURSOR>(
      LoadImage(instance, MAKEINTRESOURCE(left), IMAGE_CURSOR, 0, 0, 0)));
  EXPECT_TRUE(cursor != NULL);

  // Convert `cursor` to `mouse_shape`.
  HDC dc = GetDC(NULL);
  std::unique_ptr<MouseCursor> mouse_shape(
      CreateMouseCursorFromHCursor(dc, cursor));
  ReleaseDC(NULL, dc);

  EXPECT_TRUE(mouse_shape.get());

  // Load `right`.
  cursor.Set(reinterpret_cast<HCURSOR>(
      LoadImage(instance, MAKEINTRESOURCE(right), IMAGE_CURSOR, 0, 0, 0)));

  ICONINFO iinfo;
  EXPECT_TRUE(GetIconInfo(cursor, &iinfo));
  EXPECT_TRUE(iinfo.hbmColor);

  // Make sure the bitmaps will be freed.
  win::ScopedBitmap scoped_mask(iinfo.hbmMask);
  win::ScopedBitmap scoped_color(iinfo.hbmColor);

  // Get `scoped_color` dimensions.
  BITMAP bitmap_info;
  EXPECT_TRUE(GetObject(scoped_color, sizeof(bitmap_info), &bitmap_info));

  int width = bitmap_info.bmWidth;
  int height = bitmap_info.bmHeight;
  EXPECT_TRUE(DesktopSize(width, height).equals(mouse_shape->image()->size()));

  // Get the pixels from `scoped_color`.
  int size = width * height;
  std::unique_ptr<uint32_t[]> data(new uint32_t[size]);
  EXPECT_TRUE(GetBitmapBits(scoped_color, size * sizeof(uint32_t), data.get()));

  // Compare the 32bpp image in `mouse_shape` with the one loaded from `right`.
  return memcmp(data.get(), mouse_shape->image()->data(),
                size * sizeof(uint32_t)) == 0;
}

}  // namespace

TEST(MouseCursorTest, MatchCursors) {
  EXPECT_TRUE(
      ConvertToMouseShapeAndCompare(IDD_CURSOR1_24BPP, IDD_CURSOR1_32BPP));

  EXPECT_TRUE(
      ConvertToMouseShapeAndCompare(IDD_CURSOR1_8BPP, IDD_CURSOR1_32BPP));

  EXPECT_TRUE(
      ConvertToMouseShapeAndCompare(IDD_CURSOR2_1BPP, IDD_CURSOR2_32BPP));

  EXPECT_TRUE(
      ConvertToMouseShapeAndCompare(IDD_CURSOR3_4BPP, IDD_CURSOR3_32BPP));
}

}  // namespace webrtc
