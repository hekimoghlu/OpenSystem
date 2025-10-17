/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#include "test/gtest.h"

namespace webrtc {

class ScreenCapturerHelperTest : public ::testing::Test {
 protected:
  ScreenCapturerHelper capturer_helper_;
};

TEST_F(ScreenCapturerHelperTest, ClearInvalidRegion) {
  DesktopRegion region(DesktopRect::MakeXYWH(1, 2, 3, 4));
  capturer_helper_.InvalidateRegion(region);
  capturer_helper_.ClearInvalidRegion();
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(region.is_empty());
}

TEST_F(ScreenCapturerHelperTest, InvalidateRegion) {
  DesktopRegion region;
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(region.is_empty());

  region.SetRect(DesktopRect::MakeXYWH(1, 2, 3, 4));
  capturer_helper_.InvalidateRegion(region);
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(1, 2, 3, 4)).Equals(region));

  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(1, 2, 3, 4)));
  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(4, 2, 3, 4)));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(1, 2, 6, 4)).Equals(region));
}

TEST_F(ScreenCapturerHelperTest, InvalidateScreen) {
  DesktopRegion region;
  capturer_helper_.InvalidateScreen(DesktopSize(12, 34));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeWH(12, 34)).Equals(region));
}

TEST_F(ScreenCapturerHelperTest, SizeMostRecent) {
  EXPECT_TRUE(capturer_helper_.size_most_recent().is_empty());
  capturer_helper_.set_size_most_recent(DesktopSize(12, 34));
  EXPECT_TRUE(DesktopSize(12, 34).equals(capturer_helper_.size_most_recent()));
}

TEST_F(ScreenCapturerHelperTest, SetLogGridSize) {
  capturer_helper_.set_size_most_recent(DesktopSize(10, 10));

  DesktopRegion region;
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion().Equals(region));

  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)).Equals(region));

  capturer_helper_.SetLogGridSize(-1);
  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)).Equals(region));

  capturer_helper_.SetLogGridSize(0);
  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)).Equals(region));

  capturer_helper_.SetLogGridSize(1);
  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)));
  capturer_helper_.TakeInvalidRegion(&region);

  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(6, 6, 2, 2)).Equals(region));

  capturer_helper_.SetLogGridSize(2);
  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(4, 4, 4, 4)).Equals(region));

  capturer_helper_.SetLogGridSize(0);
  capturer_helper_.InvalidateRegion(
      DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)));
  capturer_helper_.TakeInvalidRegion(&region);
  EXPECT_TRUE(DesktopRegion(DesktopRect::MakeXYWH(7, 7, 1, 1)).Equals(region));
}

void TestExpandRegionToGrid(const DesktopRegion& region,
                            int log_grid_size,
                            const DesktopRegion& expanded_region_expected) {
  DesktopRegion expanded_region1;
  ScreenCapturerHelper::ExpandToGrid(region, log_grid_size, &expanded_region1);
  EXPECT_TRUE(expanded_region_expected.Equals(expanded_region1));

  DesktopRegion expanded_region2;
  ScreenCapturerHelper::ExpandToGrid(expanded_region1, log_grid_size,
                                     &expanded_region2);
  EXPECT_TRUE(expanded_region1.Equals(expanded_region2));
}

void TestExpandRectToGrid(int l,
                          int t,
                          int r,
                          int b,
                          int log_grid_size,
                          int lExpanded,
                          int tExpanded,
                          int rExpanded,
                          int bExpanded) {
  TestExpandRegionToGrid(DesktopRegion(DesktopRect::MakeLTRB(l, t, r, b)),
                         log_grid_size,
                         DesktopRegion(DesktopRect::MakeLTRB(
                             lExpanded, tExpanded, rExpanded, bExpanded)));
}

TEST_F(ScreenCapturerHelperTest, ExpandToGrid) {
  const int kLogGridSize = 4;
  const int kGridSize = 1 << kLogGridSize;
  for (int i = -2; i <= 2; i++) {
    int x = i * kGridSize;
    for (int j = -2; j <= 2; j++) {
      int y = j * kGridSize;
      TestExpandRectToGrid(x + 0, y + 0, x + 1, y + 1, kLogGridSize, x + 0,
                           y + 0, x + kGridSize, y + kGridSize);
      TestExpandRectToGrid(x + 0, y + kGridSize - 1, x + 1, y + kGridSize,
                           kLogGridSize, x + 0, y + 0, x + kGridSize,
                           y + kGridSize);
      TestExpandRectToGrid(x + kGridSize - 1, y + kGridSize - 1, x + kGridSize,
                           y + kGridSize, kLogGridSize, x + 0, y + 0,
                           x + kGridSize, y + kGridSize);
      TestExpandRectToGrid(x + kGridSize - 1, y + 0, x + kGridSize, y + 1,
                           kLogGridSize, x + 0, y + 0, x + kGridSize,
                           y + kGridSize);
      TestExpandRectToGrid(x - 1, y + 0, x + 1, y + 1, kLogGridSize,
                           x - kGridSize, y + 0, x + kGridSize, y + kGridSize);
      TestExpandRectToGrid(x - 1, y - 1, x + 1, y + 0, kLogGridSize,
                           x - kGridSize, y - kGridSize, x + kGridSize, y);
      TestExpandRectToGrid(x + 0, y - 1, x + 1, y + 1, kLogGridSize, x,
                           y - kGridSize, x + kGridSize, y + kGridSize);
      TestExpandRectToGrid(x - 1, y - 1, x + 0, y + 1, kLogGridSize,
                           x - kGridSize, y - kGridSize, x, y + kGridSize);

      // Construct a region consisting of 3 pixels and verify that it's expanded
      // properly to 3 squares that are kGridSize by kGridSize.
      for (int q = 0; q < 4; ++q) {
        DesktopRegion region;
        DesktopRegion expanded_region_expected;

        if (q != 0) {
          region.AddRect(DesktopRect::MakeXYWH(x - 1, y - 1, 1, 1));
          expanded_region_expected.AddRect(DesktopRect::MakeXYWH(
              x - kGridSize, y - kGridSize, kGridSize, kGridSize));
        }
        if (q != 1) {
          region.AddRect(DesktopRect::MakeXYWH(x, y - 1, 1, 1));
          expanded_region_expected.AddRect(
              DesktopRect::MakeXYWH(x, y - kGridSize, kGridSize, kGridSize));
        }
        if (q != 2) {
          region.AddRect(DesktopRect::MakeXYWH(x - 1, y, 1, 1));
          expanded_region_expected.AddRect(
              DesktopRect::MakeXYWH(x - kGridSize, y, kGridSize, kGridSize));
        }
        if (q != 3) {
          region.AddRect(DesktopRect::MakeXYWH(x, y, 1, 1));
          expanded_region_expected.AddRect(
              DesktopRect::MakeXYWH(x, y, kGridSize, kGridSize));
        }

        TestExpandRegionToGrid(region, kLogGridSize, expanded_region_expected);
      }
    }
  }
}

}  // namespace webrtc
