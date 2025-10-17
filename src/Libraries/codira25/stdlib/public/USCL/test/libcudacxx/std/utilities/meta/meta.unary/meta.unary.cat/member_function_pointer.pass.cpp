/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

// member_function_pointer

#include <uscl/std/type_traits>

#include "test_macros.h"

// NOTE: On Windows the function `test_is_member_function<void()>` and
// `test_is_member_function<void() noexcept> has the same mangled despite being
// a distinct instantiation. This causes Clang to emit an error. However
// structs do not have this problem.
template <class T>
struct test_member_function_pointer_imp
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
  static_assert(cuda::std::is_member_function_pointer<T>::value, "");
  static_assert(!cuda::std::is_enum<T>::value, "");
  static_assert(!cuda::std::is_union<T>::value, "");
  static_assert(!cuda::std::is_class<T>::value, "");
  static_assert(!cuda::std::is_function<T>::value, "");
};

template <class T>
struct test_member_function_pointer
    : test_member_function_pointer_imp<T>
    , test_member_function_pointer_imp<const T>
    , test_member_function_pointer_imp<volatile T>
    , test_member_function_pointer_imp<const volatile T>
{};

class Class
{};

struct incomplete_type;

int main(int, char**)
{
  test_member_function_pointer<void (Class::*)()>();
  test_member_function_pointer<void (Class::*)(int)>();
  test_member_function_pointer<void (Class::*)(int, char)>();

  test_member_function_pointer<void (Class::*)() const>();
  test_member_function_pointer<void (Class::*)(int) const>();
  test_member_function_pointer<void (Class::*)(int, char) const>();

  test_member_function_pointer<void (Class::*)() volatile>();
  test_member_function_pointer<void (Class::*)(int) volatile>();
  test_member_function_pointer<void (Class::*)(int, char) volatile>();

  test_member_function_pointer<void (Class::*)(...)>();
  test_member_function_pointer<void (Class::*)(int, ...)>();
  test_member_function_pointer<void (Class::*)(int, char, ...)>();

  test_member_function_pointer<void (Class::*)(...) const>();
  test_member_function_pointer<void (Class::*)(int, ...) const>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const>();

  test_member_function_pointer<void (Class::*)(...) volatile>();
  test_member_function_pointer<void (Class::*)(int, ...) volatile>();
  test_member_function_pointer<void (Class::*)(int, char, ...) volatile>();

  // reference qualifiers on functions are a C++11 extension
  // Noexcept qualifiers
  test_member_function_pointer<void (Class::*)() noexcept>();
  test_member_function_pointer<void (Class::*)(int) noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) noexcept>();

  test_member_function_pointer<void (Class::*)() const noexcept>();
  test_member_function_pointer<void (Class::*)(int) const noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) const noexcept>();

  test_member_function_pointer<void (Class::*)() volatile noexcept>();
  test_member_function_pointer<void (Class::*)(int) volatile noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) volatile noexcept>();

  test_member_function_pointer<void (Class::*)(...) noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) noexcept>();

  test_member_function_pointer<void (Class::*)(...) const noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) const noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const noexcept>();

  test_member_function_pointer<void (Class::*)(...) volatile noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) volatile noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) volatile noexcept>();

  // lvalue qualifiers
  test_member_function_pointer<void (Class::*)() &>();
  test_member_function_pointer<void (Class::*)(int) &>();
  test_member_function_pointer<void (Class::*)(int, char) &>();
  test_member_function_pointer<void (Class::*)(...) &>();
  test_member_function_pointer<void (Class::*)(int, ...) &>();
  test_member_function_pointer<void (Class::*)(int, char, ...) &>();

  test_member_function_pointer<void (Class::*)() const&>();
  test_member_function_pointer<void (Class::*)(int) const&>();
  test_member_function_pointer<void (Class::*)(int, char) const&>();
  test_member_function_pointer<void (Class::*)(...) const&>();
  test_member_function_pointer<void (Class::*)(int, ...) const&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const&>();

  test_member_function_pointer<void (Class::*)() volatile&>();
  test_member_function_pointer<void (Class::*)(int) volatile&>();
  test_member_function_pointer<void (Class::*)(int, char) volatile&>();
  test_member_function_pointer<void (Class::*)(...) volatile&>();
  test_member_function_pointer<void (Class::*)(int, ...) volatile&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) volatile&>();

  test_member_function_pointer<void (Class::*)() const volatile&>();
  test_member_function_pointer<void (Class::*)(int) const volatile&>();
  test_member_function_pointer<void (Class::*)(int, char) const volatile&>();
  test_member_function_pointer<void (Class::*)(...) const volatile&>();
  test_member_function_pointer<void (Class::*)(int, ...) const volatile&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const volatile&>();

  // Lvalue qualifiers with noexcept
  test_member_function_pointer<void (Class::*)() & noexcept>();
  test_member_function_pointer<void (Class::*)(int) & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) & noexcept>();
  test_member_function_pointer<void (Class::*)(...) & noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) & noexcept>();

  test_member_function_pointer<void (Class::*)() const & noexcept>();
  test_member_function_pointer<void (Class::*)(int) const & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) const & noexcept>();
  test_member_function_pointer<void (Class::*)(...) const & noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) const & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const & noexcept>();

  test_member_function_pointer<void (Class::*)() volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int) volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(...) volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) volatile & noexcept>();

  test_member_function_pointer<void (Class::*)() const volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int) const volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) const volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(...) const volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) const volatile & noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const volatile & noexcept>();

  // RValue qualifiers
  test_member_function_pointer<void (Class::*)() &&>();
  test_member_function_pointer<void (Class::*)(int) &&>();
  test_member_function_pointer<void (Class::*)(int, char) &&>();
  test_member_function_pointer<void (Class::*)(...) &&>();
  test_member_function_pointer<void (Class::*)(int, ...) &&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) &&>();

  test_member_function_pointer<void (Class::*)() const&&>();
  test_member_function_pointer<void (Class::*)(int) const&&>();
  test_member_function_pointer<void (Class::*)(int, char) const&&>();
  test_member_function_pointer<void (Class::*)(...) const&&>();
  test_member_function_pointer<void (Class::*)(int, ...) const&&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const&&>();

  test_member_function_pointer<void (Class::*)() volatile&&>();
  test_member_function_pointer<void (Class::*)(int) volatile&&>();
  test_member_function_pointer<void (Class::*)(int, char) volatile&&>();
  test_member_function_pointer<void (Class::*)(...) volatile&&>();
  test_member_function_pointer<void (Class::*)(int, ...) volatile&&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) volatile&&>();

  test_member_function_pointer<void (Class::*)() const volatile&&>();
  test_member_function_pointer<void (Class::*)(int) const volatile&&>();
  test_member_function_pointer<void (Class::*)(int, char) const volatile&&>();
  test_member_function_pointer<void (Class::*)(...) const volatile&&>();
  test_member_function_pointer<void (Class::*)(int, ...) const volatile&&>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const volatile&&>();

  // RValue qualifiers with noexcept
  test_member_function_pointer<void (Class::*)() && noexcept>();
  test_member_function_pointer<void (Class::*)(int) && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) && noexcept>();
  test_member_function_pointer<void (Class::*)(...) && noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) && noexcept>();

  test_member_function_pointer<void (Class::*)() const && noexcept>();
  test_member_function_pointer<void (Class::*)(int) const && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) const && noexcept>();
  test_member_function_pointer<void (Class::*)(...) const && noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) const && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const && noexcept>();

  test_member_function_pointer<void (Class::*)() volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int) volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(...) volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) volatile && noexcept>();

  test_member_function_pointer<void (Class::*)() const volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int) const volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char) const volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(...) const volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int, ...) const volatile && noexcept>();
  test_member_function_pointer<void (Class::*)(int, char, ...) const volatile && noexcept>();

  //  LWG#2582
  static_assert(!cuda::std::is_member_function_pointer<incomplete_type>::value, "");

  return 0;
}
