/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "sdk/objc/helpers/scoped_cftyperef.h"

#include "test/gtest.h"

namespace {
struct TestType {
  TestType() : has_value(true) {}
  TestType(bool b) : has_value(b) {}
  operator bool() { return has_value; }
  bool has_value;
  int retain_count = 0;
};

typedef TestType* TestTypeRef;

struct TestTypeTraits {
  static TestTypeRef InvalidValue() { return TestTypeRef(false); }
  static void Release(TestTypeRef t) { t->retain_count--; }
  static TestTypeRef Retain(TestTypeRef t) {
    t->retain_count++;
    return t;
  }
};
}  // namespace

using ScopedTestType = rtc::internal::ScopedTypeRef<TestTypeRef, TestTypeTraits>;

// In these tests we sometime introduce variables just to
// observe side-effects. Ignore the compilers complaints.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"

TEST(ScopedTypeRefTest, ShouldNotRetainByDefault) {
  TestType a;
  ScopedTestType ref(&a);
  EXPECT_EQ(0, a.retain_count);
}

TEST(ScopedTypeRefTest, ShouldRetainWithPolicy) {
  TestType a;
  ScopedTestType ref(&a, rtc::RetainPolicy::RETAIN);
  EXPECT_EQ(1, a.retain_count);
}

TEST(ScopedTypeRefTest, ShouldReleaseWhenLeavingScope) {
  TestType a;
  EXPECT_EQ(0, a.retain_count);
  {
    ScopedTestType ref(&a, rtc::RetainPolicy::RETAIN);
    EXPECT_EQ(1, a.retain_count);
  }
  EXPECT_EQ(0, a.retain_count);
}

TEST(ScopedTypeRefTest, ShouldBeCopyable) {
  TestType a;
  EXPECT_EQ(0, a.retain_count);
  {
    ScopedTestType ref1(&a, rtc::RetainPolicy::RETAIN);
    EXPECT_EQ(1, a.retain_count);
    ScopedTestType ref2 = ref1;
    EXPECT_EQ(2, a.retain_count);
  }
  EXPECT_EQ(0, a.retain_count);
}

TEST(ScopedTypeRefTest, CanReleaseOwnership) {
  TestType a;
  EXPECT_EQ(0, a.retain_count);
  {
    ScopedTestType ref(&a, rtc::RetainPolicy::RETAIN);
    EXPECT_EQ(1, a.retain_count);
    TestTypeRef b = ref.release();
  }
  EXPECT_EQ(1, a.retain_count);
}

TEST(ScopedTypeRefTest, ShouldBeTestableForTruthiness) {
  ScopedTestType ref;
  EXPECT_FALSE(ref);
  TestType a;
  ref = &a;
  EXPECT_TRUE(ref);
  ref.release();
  EXPECT_FALSE(ref);
}

TEST(ScopedTypeRefTest, ShouldProvideAccessToWrappedType) {
  TestType a;
  ScopedTestType ref(&a);
  EXPECT_EQ(&(a.retain_count), &(ref->retain_count));
}

#pragma clang diagnostic pop
