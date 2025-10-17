/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// function

#include <uscl/std/type_traits>

#include "test_macros.h"

using namespace std;

class Class
{};

enum Enum1
{
};
enum class Enum2 : int
{
};

template <class T>
__host__ __device__ void test()
{
  static_assert(!cuda::std::is_void<T>::value, "");
  static_assert(!cuda::std::is_null_pointer<T>::value, "");
  static_assert(!cuda::std::is_integral<T>::value, "");
  static_assert(!cuda::std::is_floating_point<T>::value, "");
  static_assert(!cuda::std::is_array<T>::value, "");
  static_assert(!cuda::std::is_pointer<T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference<T>::value, "");
  static_assert(!cuda::std::is_rvalue_reference<T>::value, "");
  static_assert(!cuda::std::is_member_object_pointer<T>::value, "");
  static_assert(!cuda::std::is_member_function_pointer<T>::value, "");
  static_assert(!cuda::std::is_enum<T>::value, "");
  static_assert(!cuda::std::is_union<T>::value, "");
  static_assert(!cuda::std::is_class<T>::value, "");
  static_assert(cuda::std::is_function<T>::value, "");
}

// Since we can't actually add the const volatile and ref qualifiers once
// later let's use a macro to do it.
#define TEST_REGULAR(...)       \
  test<__VA_ARGS__>();          \
  test<__VA_ARGS__ const>();    \
  test<__VA_ARGS__ volatile>(); \
  test<__VA_ARGS__ const volatile>()

#define TEST_REF_QUALIFIED(...)        \
  test<__VA_ARGS__&>();                \
  test<__VA_ARGS__ const&>();          \
  test<__VA_ARGS__ volatile&>();       \
  test<__VA_ARGS__ const volatile&>(); \
  test<__VA_ARGS__&&>();               \
  test<__VA_ARGS__ const&&>();         \
  test<__VA_ARGS__ volatile&&>();      \
  test<__VA_ARGS__ const volatile&&>()

struct incomplete_type;

int main(int, char**)
{
  TEST_REGULAR(void());
  TEST_REGULAR(void(int));
  TEST_REGULAR(int(double));
  TEST_REGULAR(int(double, char));
  TEST_REGULAR(void(...));
  TEST_REGULAR(void(int, ...));
  TEST_REGULAR(int(double, ...));
  TEST_REGULAR(int(double, char, ...));
  TEST_REF_QUALIFIED(void());
  TEST_REF_QUALIFIED(void(int));
  TEST_REF_QUALIFIED(int(double));
  TEST_REF_QUALIFIED(int(double, char));
  TEST_REF_QUALIFIED(void(...));
  TEST_REF_QUALIFIED(void(int, ...));
  TEST_REF_QUALIFIED(int(double, ...));
  TEST_REF_QUALIFIED(int(double, char, ...));

  //  LWG#2582
  static_assert(!cuda::std::is_function<incomplete_type>::value, "");

  return 0;
}
