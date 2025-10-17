/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "rtc_base/memory/always_valid_pointer.h"

#include <string>

#include "test/gtest.h"

namespace webrtc {

TEST(AlwaysValidPointerTest, DefaultToEmptyValue) {
  AlwaysValidPointer<std::string> ptr(nullptr);
  EXPECT_EQ(*ptr, "");
}
TEST(AlwaysValidPointerTest, DefaultWithForwardedArgument) {
  AlwaysValidPointer<std::string> ptr(nullptr, "test");
  EXPECT_EQ(*ptr, "test");
}
TEST(AlwaysValidPointerTest, DefaultToSubclass) {
  struct A {
    virtual ~A() {}
    virtual int f() = 0;
  };
  struct B : public A {
    int b = 0;
    explicit B(int val) : b(val) {}
    virtual ~B() {}
    int f() override { return b; }
  };
  AlwaysValidPointer<A, B> ptr(nullptr, 3);
  EXPECT_EQ(ptr->f(), 3);
  EXPECT_EQ((*ptr).f(), 3);
  EXPECT_EQ(ptr.get()->f(), 3);
}
TEST(AlwaysValidPointerTest, NonDefaultValue) {
  std::string str("keso");
  AlwaysValidPointer<std::string> ptr(&str, "test");
  EXPECT_EQ(*ptr, "keso");
}

TEST(AlwaysValidPointerTest, TakeOverOwnershipOfInstance) {
  std::string str("keso");
  std::unique_ptr<std::string> str2 = std::make_unique<std::string>("kent");
  AlwaysValidPointer<std::string> ptr(std::move(str2), &str);
  EXPECT_EQ(*ptr, "kent");
  EXPECT_EQ(str2, nullptr);
}

TEST(AlwaysValidPointerTest, TakeOverOwnershipFallbackOnPointer) {
  std::string str("keso");
  std::unique_ptr<std::string> str2;
  AlwaysValidPointer<std::string> ptr(std::move(str2), &str);
  EXPECT_EQ(*ptr, "keso");
}

TEST(AlwaysValidPointerTest, TakeOverOwnershipFallbackOnDefault) {
  std::unique_ptr<std::string> str;
  std::string* str_ptr = nullptr;
  AlwaysValidPointer<std::string> ptr(std::move(str), str_ptr);
  EXPECT_EQ(*ptr, "");
}

TEST(AlwaysValidPointerTest,
     TakeOverOwnershipFallbackOnDefaultWithForwardedArgument) {
  std::unique_ptr<std::string> str2;
  AlwaysValidPointer<std::string> ptr(std::move(str2), nullptr, "keso");
  EXPECT_EQ(*ptr, "keso");
}

TEST(AlwaysValidPointerTest, TakeOverOwnershipDoesNotForwardDefaultArguments) {
  std::unique_ptr<std::string> str = std::make_unique<std::string>("kalle");
  std::unique_ptr<std::string> str2 = std::make_unique<std::string>("anka");
  AlwaysValidPointer<std::string> ptr(std::move(str), nullptr, *str2);
  EXPECT_EQ(*ptr, "kalle");
  EXPECT_TRUE(!str);
  EXPECT_EQ(*str2, "anka");
}

TEST(AlwaysValidPointerTest, DefaultToLambda) {
  AlwaysValidPointer<std::string> ptr(
      nullptr, []() { return std::make_unique<std::string>("onkel skrue"); });
  EXPECT_EQ(*ptr, "onkel skrue");
}

TEST(AlwaysValidPointerTest, NoDefaultObjectPassValidPointer) {
  std::string str("foo");
  AlwaysValidPointerNoDefault<std::string> ptr(&str);
  EXPECT_EQ(*ptr, "foo");
  EXPECT_EQ(ptr, &str);
}

TEST(AlwaysValidPointerTest, NoDefaultObjectWithTakeOverOwnership) {
  std::unique_ptr<std::string> str = std::make_unique<std::string>("yum");
  AlwaysValidPointerNoDefault<std::string> ptr(std::move(str));
  EXPECT_EQ(*ptr, "yum");
  std::unique_ptr<std::string> str2 = std::make_unique<std::string>("fun");
  AlwaysValidPointerNoDefault<std::string> ptr2(std::move(str), str2.get());
  EXPECT_EQ(*ptr2, "fun");
  EXPECT_EQ(ptr2, str2.get());
}

#if GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)

TEST(AlwaysValidPointerTest, NoDefaultObjectPassNullPointer) {
  auto pass_null = []() {
    AlwaysValidPointerNoDefault<std::string> ptr(nullptr);
  };
  EXPECT_DEATH(pass_null(), "");
}

TEST(AlwaysValidPointerTest, NoDefaultObjectPassNullUniquePointer) {
  auto pass_null = []() {
    std::unique_ptr<std::string> str;
    AlwaysValidPointerNoDefault<std::string> ptr(std::move(str));
  };
  EXPECT_DEATH(pass_null(), "");
}

#endif

}  // namespace webrtc
