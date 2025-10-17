/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// has weak result type

// UNSUPPORTED: c++20

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

//// #include <uscl/std/functional>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

template <class Arg, class Result>
struct my_unary_function
{ // cuda::std::unary_function was removed in C++17
  typedef Arg argument_type;
  typedef Result result_type;
};

template <class Arg1, class Arg2, class Result>
struct my_binary_function
{ // cuda::std::binary_function was removed in C++17
  typedef Arg1 first_argument_type;
  typedef Arg2 second_argument_type;
  typedef Result result_type;
};

class functor1 : public my_unary_function<int, char>
{};

class functor2 : public my_binary_function<char, int, double>
{};

class functor3
    : public my_unary_function<char, int>
    , public my_binary_function<char, int, double>
{
public:
  typedef float result_type;
};

class functor4
    : public my_unary_function<char, int>
    , public my_binary_function<char, int, double>
{
public:
};

class C
{};

template <class T>
struct has_result_type
{
private:
  struct two
  {
    char _;
    char __;
  };
  template <class U>
  __host__ __device__ static two test(...);
  template <class U>
  __host__ __device__ static char test(typename U::result_type* = 0);

public:
  static const bool value = sizeof(test<T>(0)) == 1;
};

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<functor1>::result_type, char>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<functor2>::result_type, double>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<functor3>::result_type, float>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<void()>::result_type, void>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<int*(double*)>::result_type, int*>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<void (*)()>::result_type, void>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<int* (*) (double*)>::result_type, int*>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<int* (C::*) (double*)>::result_type, int*>::value),
                "");
  static_assert(
    (cuda::std::is_same<cuda::std::reference_wrapper<int (C::*)(double*) const volatile>::result_type, int>::value),
    "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<C()>::result_type, C>::value), "");
  static_assert(has_result_type<cuda::std::reference_wrapper<functor3>>::value, "");
  static_assert(!has_result_type<cuda::std::reference_wrapper<functor4>>::value, "");
  static_assert(!has_result_type<cuda::std::reference_wrapper<C>>::value, "");

  return 0;
}
