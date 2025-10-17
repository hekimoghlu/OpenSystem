/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
// concept equality_comparable = // see below

#include <uscl/std/array>
#include <uscl/std/concepts>

#include "compare_types.h"

using cuda::std::equality_comparable;

namespace fundamentals
{
static_assert(equality_comparable<int>, "");
static_assert(equality_comparable<double>, "");
static_assert(equality_comparable<void*>, "");
static_assert(equality_comparable<char*>, "");
static_assert(equality_comparable<char const*>, "");
static_assert(equality_comparable<char volatile*>, "");
static_assert(equality_comparable<char const volatile*>, "");
static_assert(equality_comparable<wchar_t&>, "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(equality_comparable<char8_t const&>, "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(equality_comparable<char16_t volatile&>, "");
static_assert(equality_comparable<char32_t const volatile&>, "");
static_assert(equality_comparable<unsigned char&&>, "");
static_assert(equality_comparable<unsigned short const&&>, "");
static_assert(equality_comparable<unsigned int volatile&&>, "");
static_assert(equality_comparable<unsigned long const volatile&&>, "");
static_assert(equality_comparable<int[5]>, "");
static_assert(equality_comparable<int (*)(int)>, "");
static_assert(equality_comparable<int (&)(int)>, "");
static_assert(equality_comparable<int (*)(int) noexcept>, "");
static_assert(equality_comparable<int (&)(int) noexcept>, "");
static_assert(equality_comparable<cuda::std::nullptr_t>, "");

struct S
{};
static_assert(equality_comparable<int S::*>, "");
static_assert(equality_comparable<int (S::*)()>, "");
static_assert(equality_comparable<int (S::*)() noexcept>, "");
static_assert(equality_comparable<int (S::*)() &>, "");
static_assert(equality_comparable<int (S::*)() & noexcept>, "");
static_assert(equality_comparable<int (S::*)() &&>, "");
static_assert(equality_comparable < int (S::*)() && noexcept >, "");
static_assert(equality_comparable<int (S::*)() const>, "");
static_assert(equality_comparable<int (S::*)() const noexcept>, "");
static_assert(equality_comparable<int (S::*)() const&>, "");
static_assert(equality_comparable<int (S::*)() const & noexcept>, "");
static_assert(equality_comparable<int (S::*)() const&&>, "");
static_assert(equality_comparable < int (S::*)() const&& noexcept >, "");
static_assert(equality_comparable<int (S::*)() volatile>, "");
static_assert(equality_comparable<int (S::*)() volatile noexcept>, "");
static_assert(equality_comparable<int (S::*)() volatile&>, "");
static_assert(equality_comparable<int (S::*)() volatile & noexcept>, "");
static_assert(equality_comparable<int (S::*)() volatile&&>, "");
static_assert(equality_comparable < int (S::*)() volatile && noexcept >, "");
static_assert(equality_comparable<int (S::*)() const volatile>, "");
static_assert(equality_comparable<int (S::*)() const volatile noexcept>, "");
static_assert(equality_comparable<int (S::*)() const volatile&>, "");
static_assert(equality_comparable<int (S::*)() const volatile & noexcept>, "");
static_assert(equality_comparable<int (S::*)() const volatile&&>, "");
static_assert(equality_comparable < int (S::*)() const volatile&& noexcept >, "");

static_assert(!equality_comparable<void>, "");
} // namespace fundamentals

namespace standard_types
{
static_assert(equality_comparable<cuda::std::array<int, 10>>, "");
} // namespace standard_types

namespace types_fit_for_purpose
{
#if TEST_STD_VER > 2017
static_assert(equality_comparable<cxx20_member_eq>, "");
static_assert(equality_comparable<cxx20_friend_eq>, "");
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(equality_comparable<member_three_way_comparable>, "");
#    if !TEST_CUDA_COMPILER(NVCC) // nvbug3908399
static_assert(equality_comparable<friend_three_way_comparable>, "");
#    endif // !__NVCC_
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(equality_comparable<explicit_operators>, "");
static_assert(equality_comparable<different_return_types>, "");
static_assert(equality_comparable<one_member_one_friend>, "");
static_assert(equality_comparable<equality_comparable_with_ec1>, "");
#endif // TEST_STD_VER > 2017

static_assert(!equality_comparable<no_eq>, "");
static_assert(!equality_comparable<no_neq>, "");
static_assert(equality_comparable<no_lt>, "");
static_assert(equality_comparable<no_gt>, "");
static_assert(equality_comparable<no_le>, "");
static_assert(equality_comparable<no_ge>, "");

static_assert(!equality_comparable<wrong_return_type_eq>, "");
static_assert(!equality_comparable<wrong_return_type_ne>, "");
static_assert(equality_comparable<wrong_return_type_lt>, "");
static_assert(equality_comparable<wrong_return_type_gt>, "");
static_assert(equality_comparable<wrong_return_type_le>, "");
static_assert(equality_comparable<wrong_return_type_ge>, "");
static_assert(!equality_comparable<wrong_return_type>, "");

#if TEST_STD_VER > 2017
static_assert(!equality_comparable<cxx20_member_eq_operator_with_deleted_ne>, "");
static_assert(!equality_comparable<cxx20_friend_eq_operator_with_deleted_ne>, "");
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(!equality_comparable<member_three_way_comparable_with_deleted_eq>, "");
static_assert(!equality_comparable<member_three_way_comparable_with_deleted_ne>, "");
static_assert(!equality_comparable<friend_three_way_comparable_with_deleted_eq>, "");
#    if !TEST_CUDA_COMPILER(NVCC) // nvbug3908399
static_assert(!equality_comparable<friend_three_way_comparable_with_deleted_ne>, "");
#    endif // !TEST_CUDA_COMPILER(NVCC)

static_assert(!equality_comparable<eq_returns_explicit_bool>, "");
static_assert(!equality_comparable<ne_returns_explicit_bool>, "");
static_assert(equality_comparable<lt_returns_explicit_bool>, "");
static_assert(equality_comparable<gt_returns_explicit_bool>, "");
static_assert(equality_comparable<le_returns_explicit_bool>, "");
static_assert(equality_comparable<ge_returns_explicit_bool>, "");
static_assert(equality_comparable<returns_true_type>, "");
static_assert(equality_comparable<returns_int_ptr>, "");
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#endif // TEST_STD_VER > 2017
} // namespace types_fit_for_purpose

int main(int, char**)
{
  return 0;
}
