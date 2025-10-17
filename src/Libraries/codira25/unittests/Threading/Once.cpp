/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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

//===--- Once.cpp - Tests the language::once() implementation ----------------===//
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

#include "language/Threading/Once.h"
#include "gtest/gtest.h"

#include <cstring>

#include "ThreadingHelpers.h"

using namespace language;

// Check that language::once calls the function, with the correct argument
TEST(OnceTest, once_calls_function) {
  static language::once_t predicate;
  bool wasCalled = false;

  language::once(
      predicate,
      [](void *ctx) {
        bool *pWasCalled = static_cast<bool *>(ctx);
        *pWasCalled = true;
      },
      &wasCalled);

  ASSERT_TRUE(wasCalled);
}

// Check that calling language::once twice only calls the function once
TEST(OnceTest, once_calls_only_once) {
  static language::once_t predicate;
  unsigned callCount = 0;

  void (*fn)(void *) = [](void *ctx) {
    unsigned *pCallCount = static_cast<unsigned *>(ctx);
    ++*pCallCount;
  };

  language::once(predicate, fn, &callCount);
  language::once(predicate, fn, &callCount);

  ASSERT_EQ(1u, callCount);
}

// Check that language::once works when threaded
TEST(OnceTest, once_threaded) {
  void (*fn)(void *) = [](void *ctx) {
    unsigned *pCallCount = static_cast<unsigned *>(ctx);
    ++*pCallCount;
  };

  for (unsigned tries = 0; tries < 1000; ++tries) {
    language::once_t predicate;
    unsigned callCount = 0;

    // We're being naughty here; language::once_t is supposed to be global/static,
    // but since we know what we're doing, this should be OK.
    std::memset(&predicate, 0, sizeof(predicate));

    threadedExecute(16, [&](int) { language::once(predicate, fn, &callCount); });

    ASSERT_EQ(1u, callCount);
  }
}

// Check that language::once works with a C++ lambda
TEST(OnceTest, once_lambda) {
  static language::once_t predicate;
  unsigned callCount = 0;

  auto fn = [&callCount]() {
    ++callCount;
  };

  language::once(predicate, fn);
  language::once(predicate, fn);

  ASSERT_EQ(1u, callCount);
}
