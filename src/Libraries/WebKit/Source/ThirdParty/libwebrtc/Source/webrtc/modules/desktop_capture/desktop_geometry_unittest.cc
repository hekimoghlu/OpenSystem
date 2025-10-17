/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

#include "test/gtest.h"

namespace webrtc {

TEST(DesktopRectTest, UnionBetweenTwoNonEmptyRects) {
  DesktopRect rect = DesktopRect::MakeLTRB(1, 1, 2, 2);
  rect.UnionWith(DesktopRect::MakeLTRB(-2, -2, -1, -1));
  ASSERT_TRUE(rect.equals(DesktopRect::MakeLTRB(-2, -2, 2, 2)));
}

TEST(DesktopRectTest, UnionWithEmptyRect) {
  DesktopRect rect = DesktopRect::MakeWH(1, 1);
  rect.UnionWith(DesktopRect());
  ASSERT_TRUE(rect.equals(DesktopRect::MakeWH(1, 1)));

  rect = DesktopRect::MakeXYWH(1, 1, 2, 2);
  rect.UnionWith(DesktopRect());
  ASSERT_TRUE(rect.equals(DesktopRect::MakeXYWH(1, 1, 2, 2)));

  rect = DesktopRect::MakeXYWH(1, 1, 2, 2);
  rect.UnionWith(DesktopRect::MakeXYWH(3, 3, 0, 0));
  ASSERT_TRUE(rect.equals(DesktopRect::MakeXYWH(1, 1, 2, 2)));
}

TEST(DesktopRectTest, EmptyRectUnionWithNonEmptyOne) {
  DesktopRect rect;
  rect.UnionWith(DesktopRect::MakeWH(1, 1));
  ASSERT_TRUE(rect.equals(DesktopRect::MakeWH(1, 1)));

  rect = DesktopRect();
  rect.UnionWith(DesktopRect::MakeXYWH(1, 1, 2, 2));
  ASSERT_TRUE(rect.equals(DesktopRect::MakeXYWH(1, 1, 2, 2)));

  rect = DesktopRect::MakeXYWH(3, 3, 0, 0);
  rect.UnionWith(DesktopRect::MakeXYWH(1, 1, 2, 2));
  ASSERT_TRUE(rect.equals(DesktopRect::MakeXYWH(1, 1, 2, 2)));
}

TEST(DesktopRectTest, EmptyRectUnionWithEmptyOne) {
  DesktopRect rect;
  rect.UnionWith(DesktopRect());
  ASSERT_TRUE(rect.is_empty());

  rect = DesktopRect::MakeXYWH(1, 1, 0, 0);
  rect.UnionWith(DesktopRect());
  ASSERT_TRUE(rect.is_empty());

  rect = DesktopRect();
  rect.UnionWith(DesktopRect::MakeXYWH(1, 1, 0, 0));
  ASSERT_TRUE(rect.is_empty());

  rect = DesktopRect::MakeXYWH(1, 1, 0, 0);
  rect.UnionWith(DesktopRect::MakeXYWH(-1, -1, 0, 0));
  ASSERT_TRUE(rect.is_empty());
}

TEST(DesktopRectTest, Scale) {
  DesktopRect rect = DesktopRect::MakeXYWH(100, 100, 100, 100);
  rect.Scale(1.1, 1.1);
  ASSERT_EQ(rect.top(), 100);
  ASSERT_EQ(rect.left(), 100);
  ASSERT_EQ(rect.width(), 110);
  ASSERT_EQ(rect.height(), 110);

  rect = DesktopRect::MakeXYWH(100, 100, 100, 100);
  rect.Scale(0.01, 0.01);
  ASSERT_EQ(rect.top(), 100);
  ASSERT_EQ(rect.left(), 100);
  ASSERT_EQ(rect.width(), 1);
  ASSERT_EQ(rect.height(), 1);

  rect = DesktopRect::MakeXYWH(100, 100, 100, 100);
  rect.Scale(1.1, 0.9);
  ASSERT_EQ(rect.top(), 100);
  ASSERT_EQ(rect.left(), 100);
  ASSERT_EQ(rect.width(), 110);
  ASSERT_EQ(rect.height(), 90);

  rect = DesktopRect::MakeXYWH(0, 0, 100, 100);
  rect.Scale(1.1, 1.1);
  ASSERT_EQ(rect.top(), 0);
  ASSERT_EQ(rect.left(), 0);
  ASSERT_EQ(rect.width(), 110);
  ASSERT_EQ(rect.height(), 110);

  rect = DesktopRect::MakeXYWH(0, 100, 100, 100);
  rect.Scale(1.1, 1.1);
  ASSERT_EQ(rect.top(), 100);
  ASSERT_EQ(rect.left(), 0);
  ASSERT_EQ(rect.width(), 110);
  ASSERT_EQ(rect.height(), 110);
}

}  // namespace webrtc
