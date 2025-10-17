/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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

//===--- TypeLookupError.cpp - TypeLookupError Tests ----------------------===//
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

#include "language/Demangling/TypeLookupError.h"
#include "gtest/gtest.h"
#include <vector>

using namespace language;

TEST(TypeLookupError, ConstantString) {
  auto error = TypeLookupError("testing testing");
  char *str = error.copyErrorString();
  ASSERT_STREQ(str, "testing testing");
  error.freeErrorString(str);
}

TEST(TypeLookupError, FormatString) {
  auto error = TYPE_LOOKUP_ERROR_FMT("%d %d %d %d %d %d %d %d %d %d", 0, 1, 2,
                                     3, 4, 5, 6, 7, 8, 9);
  char *str = error.copyErrorString();
  ASSERT_STREQ(str, "0 1 2 3 4 5 6 7 8 9");
  error.freeErrorString(str);
}

TEST(TypeLookupError, Copying) {
  std::vector<TypeLookupError> vec;

  {
    auto originalError = TYPE_LOOKUP_ERROR_FMT("%d %d %d %d %d %d %d %d %d %d",
                                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    for (int i = 0; i < 5; i++)
      vec.push_back(originalError);
  }

  for (auto &error : vec) {
    char *str = error.copyErrorString();
    ASSERT_STREQ(str, "0 1 2 3 4 5 6 7 8 9");
    error.freeErrorString(str);
  }

  auto extractedError = vec[4];
  vec.clear();
  char *str = extractedError.copyErrorString();
  ASSERT_STREQ(str, "0 1 2 3 4 5 6 7 8 9");
  extractedError.freeErrorString(str);
}
