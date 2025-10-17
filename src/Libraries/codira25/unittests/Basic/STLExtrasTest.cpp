/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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

//===--- STLExtrasTest.cpp ------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "language/Basic/STLExtras.h"
#include "gtest/gtest.h"

using namespace language;

TEST(RemoveAdjacentIf, NoRemovals) {
  {
    int items[] = { 1, 2, 3 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, std::end(items));
  }

  {
    int items[] = { 1 };
    // Test an empty range.
    auto result = removeAdjacentIf(std::begin(items), std::begin(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, std::begin(items));
  }

  {
    int *null = nullptr;
    auto result = removeAdjacentIf(null, null, std::equal_to<int>());
    EXPECT_EQ(result, null);
  }
}

TEST(RemoveAdjacentIf, OnlyOneRun) {
  {
    int items[] = { 1, 2, 3, 3, 4, 5, 6 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[5]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
    EXPECT_EQ(items[2], 4);
    EXPECT_EQ(items[3], 5);
    EXPECT_EQ(items[4], 6);
  }

  {
    int items[] = { 1, 2, 3, 3, 3, 4, 5, 6 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[5]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
    EXPECT_EQ(items[2], 4);
    EXPECT_EQ(items[3], 5);
    EXPECT_EQ(items[4], 6);
  }

  {
    int items[] = { 1, 2, 3, 3, 3, 3, 4, 5, 6 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[5]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
    EXPECT_EQ(items[2], 4);
    EXPECT_EQ(items[3], 5);
    EXPECT_EQ(items[4], 6);
  }

  {
    int items[] = { 1, 2, 3, 3, 3, 3 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[2]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
  }

  {
    int items[] = { 3, 3, 3, 3, 4, 5, 6 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[3]);
    EXPECT_EQ(items[0], 4);
    EXPECT_EQ(items[1], 5);
    EXPECT_EQ(items[2], 6);
  }

  {
    int items[] = { 1, 1, 1, 1 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[0]);
  }
}


TEST(RemoveAdjacentIf, MultipleRuns) {
  {
    int items[] = { 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[7]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
    EXPECT_EQ(items[2], 4);
    EXPECT_EQ(items[3], 5);
    EXPECT_EQ(items[4], 6);
    EXPECT_EQ(items[5], 8);
    EXPECT_EQ(items[6], 9);
  }

  {
    int items[] = { 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 9 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[7]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
    EXPECT_EQ(items[2], 4);
    EXPECT_EQ(items[3], 5);
    EXPECT_EQ(items[4], 6);
    EXPECT_EQ(items[5], 8);
    EXPECT_EQ(items[6], 9);
  }

  {
    int items[] = { 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 9 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[7]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
    EXPECT_EQ(items[2], 4);
    EXPECT_EQ(items[3], 5);
    EXPECT_EQ(items[4], 6);
    EXPECT_EQ(items[5], 8);
    EXPECT_EQ(items[6], 9);
  }

  {
    int items[] = { 1, 2, 3, 3, 3, 3, 7, 7 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[2]);
    EXPECT_EQ(items[0], 1);
    EXPECT_EQ(items[1], 2);
  }

  {
    int items[] = { 3, 3, 3, 3, 4, 5, 6, 7, 7 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[3]);
    EXPECT_EQ(items[0], 4);
    EXPECT_EQ(items[1], 5);
    EXPECT_EQ(items[2], 6);
  }

  {
    int items[] = { 3, 3, 3, 3, 7, 7, 8, 9 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[2]);
    EXPECT_EQ(items[0], 8);
    EXPECT_EQ(items[1], 9);
  }

  {
    int items[] = { 1, 1, 1, 1, 2, 2 };
    auto result = removeAdjacentIf(std::begin(items), std::end(items),
                                   std::equal_to<int>());
    EXPECT_EQ(result, &items[0]);
  }
}
