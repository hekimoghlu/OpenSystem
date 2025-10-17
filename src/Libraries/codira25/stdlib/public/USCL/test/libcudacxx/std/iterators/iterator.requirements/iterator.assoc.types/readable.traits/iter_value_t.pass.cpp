/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

// template<class T>
// using iter_value_t;

#include <uscl/std/concepts>
#include <uscl/std/iterator>

template <class T, class Expected>
__host__ __device__ constexpr bool check_iter_value_t()
{
  constexpr bool result = cuda::std::same_as<cuda::std::iter_value_t<T>, Expected>;
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T volatile>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const volatile>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const&>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T volatile&>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const volatile&>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const&&>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T volatile&&>, Expected> == result, "");
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const volatile&&>, Expected> == result, "");

  return result;
}

static_assert(check_iter_value_t<int*, int>(), "");
static_assert(check_iter_value_t<int[], int>(), "");
static_assert(check_iter_value_t<int[10], int>(), "");

struct both_members
{
  using value_type   = double;
  using element_type = double;
};
static_assert(check_iter_value_t<both_members, double>(), "");

// clang-format off
template <class T, class = void>
inline constexpr bool check_no_iter_value_t = true;

template <class T>
inline constexpr bool check_no_iter_value_t<T, cuda::std::void_t<cuda::std::iter_value_t<T>>> = false;

static_assert(check_no_iter_value_t<void>, "");
static_assert(check_no_iter_value_t<double>, "");

struct S {};
static_assert(check_no_iter_value_t<S>, "");

struct different_value_element_members {
  using value_type = int;
  using element_type = long;
};
static_assert(check_no_iter_value_t<different_value_element_members>, "");

int main(int, char**) { return 0; }
