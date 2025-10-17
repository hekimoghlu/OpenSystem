/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

//===--- ThreadSafeRefCntPointerTest.cpp ----------------------------------===//
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

#include "language/Basic/ThreadSafeRefCounted.h"
#include "toolchain/ADT/IntrusiveRefCntPtr.h"
#include "gtest/gtest.h"

using toolchain::IntrusiveRefCntPtr;

struct TestRelease : toolchain::ThreadSafeRefCountedBase<TestRelease> {
  bool &destroy;
  TestRelease(bool &destroy) : destroy(destroy) {}
  ~TestRelease() { destroy = true; }
};

TEST(ThreadSafeRefCountedBase, ReleaseSimple) {
  bool destroyed = false;
  {
    IntrusiveRefCntPtr<TestRelease> ref = new TestRelease(destroyed);
  }
  EXPECT_TRUE(destroyed);
}
TEST(ThreadSafeRefCountedBase, Release) {
  bool destroyed = false;
  {
    IntrusiveRefCntPtr<TestRelease> ref = new TestRelease(destroyed);
    ref->Retain();
    ref->Release();
  }
  EXPECT_TRUE(destroyed);
}

struct TestReleaseVPTR : language::ThreadSafeRefCountedBaseVPTR {
  bool &destroy;
  TestReleaseVPTR(bool &destroy) : destroy(destroy) {}
  ~TestReleaseVPTR() override { destroy = true; }
};

TEST(ThreadSafeRefCountedBaseVPTR, ReleaseSimple) {
  bool destroyed = false;
  {
    IntrusiveRefCntPtr<TestReleaseVPTR> ref = new TestReleaseVPTR(destroyed);
  }
  EXPECT_TRUE(destroyed);
}
TEST(ThreadSafeRefCountedBaseVPTR, Release) {
  bool destroyed = false;
  {
    IntrusiveRefCntPtr<TestReleaseVPTR> ref = new TestReleaseVPTR(destroyed);
    ref->Retain();
    ref->Release();
  }
  EXPECT_TRUE(destroyed);
}
