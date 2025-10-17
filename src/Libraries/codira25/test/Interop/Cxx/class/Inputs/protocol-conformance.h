/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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

#ifndef TEST_INTEROP_CXX_CLASS_INPUTS_PROTOCOL_CONFORMANCE_H
#define TEST_INTEROP_CXX_CLASS_INPUTS_PROTOCOL_CONFORMANCE_H

struct ConformsToProtocol {
  int return42() { return 42; }
};

struct DoesNotConformToProtocol {
  int returnFortyTwo() { return 42; }
};

struct DummyStruct {};

struct __attribute__((language_attr("import_unsafe"))) NonTrivial {
  NonTrivial(const NonTrivial &other) {}
  ~NonTrivial() {}
  NonTrivial(DummyStruct) {}
  NonTrivial() {}
  void test1() {}
  void test2(int) {}
  char test3(int, unsigned) { return 42; }
};

struct Trivial {
  Trivial(DummyStruct) {}
  Trivial() {}
  void test1() {}
  void test2(int) {}
  char test3(int, unsigned) { return 42; }
};

struct ReturnsNullableValue {
  const int *returnPointer() __attribute__((language_attr("import_unsafe"))) {
    return nullptr;
  }
};

struct ReturnsNonNullValue {
  const int *returnPointer() __attribute__((returns_nonnull))
  __attribute__((language_attr("import_unsafe"))) {
    return (int *)this;
  }
};

struct HasOperatorExclaim {
  int value;

  HasOperatorExclaim operator!() const { return {-value}; }
};

struct HasOperatorEqualEqual {
  int value;
  
  bool operator==(const HasOperatorEqualEqual &other) const {
    return value == other.value;
  }
};

template <typename T>
struct HasOperatorPlusEqual {
  T value;

  HasOperatorPlusEqual &operator+=(int x) {
    value += x;
    return *this;
  }
};

using HasOperatorPlusEqualInt = HasOperatorPlusEqual<int>;

struct HasVirtualMethod {
  virtual int return42() { return 42; } 
};

struct HasStaticOperatorCall {
  static int operator()(int x) { return x * 2; }
};

typedef struct {
  int a;
} Anon0;

typedef struct {
  int a;
} Anon1;

template <class T>
struct S {
  ~S() {}
  int method0();
};

using AnonType0 = S<Anon0>;
using AnonType1 = S<Anon1>;

#endif // TEST_INTEROP_CXX_CLASS_INPUTS_PROTOCOL_CONFORMANCE_H
