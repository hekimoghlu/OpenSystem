/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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
#include "api/scoped_refptr.h"

#include <type_traits>
#include <utility>
#include <vector>

#include "test/gtest.h"

namespace rtc {
namespace {

struct FunctionsCalled {
  int addref = 0;
  int release = 0;
};

class ScopedRefCounted {
 public:
  explicit ScopedRefCounted(FunctionsCalled* called) : called_(*called) {}
  ScopedRefCounted(const ScopedRefCounted&) = delete;
  ScopedRefCounted& operator=(const ScopedRefCounted&) = delete;

  void AddRef() {
    ++called_.addref;
    ++ref_count_;
  }
  void Release() {
    ++called_.release;
    if (0 == --ref_count_)
      delete this;
  }

 private:
  ~ScopedRefCounted() = default;

  FunctionsCalled& called_;
  int ref_count_ = 0;
};

TEST(ScopedRefptrTest, IsCopyConstructable) {
  FunctionsCalled called;
  scoped_refptr<ScopedRefCounted> ptr(new ScopedRefCounted(&called));
  scoped_refptr<ScopedRefCounted> another_ptr = ptr;

  EXPECT_TRUE(ptr);
  EXPECT_TRUE(another_ptr);
  EXPECT_EQ(called.addref, 2);
}

TEST(ScopedRefptrTest, IsCopyAssignable) {
  FunctionsCalled called;
  scoped_refptr<ScopedRefCounted> another_ptr;
  scoped_refptr<ScopedRefCounted> ptr(new ScopedRefCounted(&called));
  another_ptr = ptr;

  EXPECT_TRUE(ptr);
  EXPECT_TRUE(another_ptr);
  EXPECT_EQ(called.addref, 2);
}

TEST(ScopedRefptrTest, IsMoveConstructableWithoutExtraAddRefRelease) {
  FunctionsCalled called;
  scoped_refptr<ScopedRefCounted> ptr(new ScopedRefCounted(&called));
  scoped_refptr<ScopedRefCounted> another_ptr = std::move(ptr);

  EXPECT_FALSE(ptr);
  EXPECT_TRUE(another_ptr);
  EXPECT_EQ(called.addref, 1);
  EXPECT_EQ(called.release, 0);
}

TEST(ScopedRefptrTest, IsMoveAssignableWithoutExtraAddRefRelease) {
  FunctionsCalled called;
  scoped_refptr<ScopedRefCounted> another_ptr;
  scoped_refptr<ScopedRefCounted> ptr(new ScopedRefCounted(&called));
  another_ptr = std::move(ptr);

  EXPECT_FALSE(ptr);
  EXPECT_TRUE(another_ptr);
  EXPECT_EQ(called.addref, 1);
  EXPECT_EQ(called.release, 0);
}

TEST(ScopedRefptrTest, MovableDuringVectorReallocation) {
  static_assert(
      std::is_nothrow_move_constructible<scoped_refptr<ScopedRefCounted>>(),
      "");
  // Test below describes a scenario where it is helpful for move constructor
  // to be noexcept.
  FunctionsCalled called;
  std::vector<scoped_refptr<ScopedRefCounted>> ptrs;
  ptrs.reserve(1);
  // Insert more elements than reserved to provoke reallocation.
  ptrs.emplace_back(new ScopedRefCounted(&called));
  ptrs.emplace_back(new ScopedRefCounted(&called));

  EXPECT_EQ(called.addref, 2);
  EXPECT_EQ(called.release, 0);
}

}  // namespace
}  // namespace rtc
