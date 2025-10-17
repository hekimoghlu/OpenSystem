/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

// template<class LHS, class RHS>
// concept assignable_from =
//   std::is_lvalue_reference_v<LHS> &&
//   std::common_reference_with<
//     const std::remove_reference_t<LHS>&,
//     const std::remove_reference_t<RHS>&> &&
//   requires (LHS lhs, RHS&& rhs) {
//     { lhs = std::forward<RHS>(rhs) } -> std::same_as<LHS>;
//   };

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "MoveOnly.h"
#include "test_macros.h"

struct NoCommonRef
{
  __host__ __device__ NoCommonRef& operator=(const int&);
};
static_assert(cuda::std::is_assignable_v<NoCommonRef&, const int&>, "");
static_assert(!cuda::std::assignable_from<NoCommonRef&, const int&>, ""); // no common reference type

struct Base
{};
struct Derived : Base
{};
static_assert(!cuda::std::assignable_from<Base*, Derived*>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived*>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived*&>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived*&&>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived* const>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived* const&>, "");
static_assert(cuda::std::assignable_from<Base*&, Derived* const&&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived*>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived*&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived*&&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived* const>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived* const&>, "");
static_assert(!cuda::std::assignable_from<Base*&, const Derived* const&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived*>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived*&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived*&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived* const>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived* const&>, "");
static_assert(cuda::std::assignable_from<const Base*&, Derived* const&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived*>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived*&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived*&&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived* const>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived* const&>, "");
static_assert(cuda::std::assignable_from<const Base*&, const Derived* const&&>, "");

struct VoidResultType
{
  __host__ __device__ void operator=(const VoidResultType&);
};
static_assert(cuda::std::is_assignable_v<VoidResultType&, const VoidResultType&>, "");
static_assert(!cuda::std::assignable_from<VoidResultType&, const VoidResultType&>, "");

struct ValueResultType
{
  __host__ __device__ ValueResultType operator=(const ValueResultType&);
};
static_assert(cuda::std::is_assignable_v<ValueResultType&, const ValueResultType&>, "");
static_assert(!cuda::std::assignable_from<ValueResultType&, const ValueResultType&>, "");

struct Locale
{
  __host__ __device__ const Locale& operator=(const Locale&);
};
static_assert(cuda::std::is_assignable_v<Locale&, const Locale&>, "");
static_assert(!cuda::std::assignable_from<Locale&, const Locale&>, "");

struct Tuple
{
  __host__ __device__ Tuple& operator=(const Tuple&);
  __host__ __device__ const Tuple& operator=(const Tuple&) const;
};
static_assert(!cuda::std::assignable_from<Tuple, const Tuple&>, "");
static_assert(cuda::std::assignable_from<Tuple&, const Tuple&>, "");
static_assert(!cuda::std::assignable_from<Tuple&&, const Tuple&>, "");
static_assert(!cuda::std::assignable_from<const Tuple, const Tuple&>, "");
static_assert(cuda::std::assignable_from<const Tuple&, const Tuple&>, "");
static_assert(!cuda::std::assignable_from<const Tuple&&, const Tuple&>, "");

// Finally, check a few simple cases.
static_assert(cuda::std::assignable_from<int&, int>, "");
static_assert(cuda::std::assignable_from<int&, int&>, "");
static_assert(cuda::std::assignable_from<int&, int&&>, "");
static_assert(!cuda::std::assignable_from<const int&, int>, "");
static_assert(!cuda::std::assignable_from<const int&, int&>, "");
static_assert(!cuda::std::assignable_from<const int&, int&&>, "");
static_assert(cuda::std::assignable_from<volatile int&, int>, "");
static_assert(cuda::std::assignable_from<volatile int&, int&>, "");
static_assert(cuda::std::assignable_from<volatile int&, int&&>, "");
static_assert(!cuda::std::assignable_from<int (&)[10], int>, "");
static_assert(!cuda::std::assignable_from<int (&)[10], int (&)[10]>, "");
static_assert(cuda::std::assignable_from<MoveOnly&, MoveOnly>, "");
static_assert(!cuda::std::assignable_from<MoveOnly&, MoveOnly&>, "");
static_assert(cuda::std::assignable_from<MoveOnly&, MoveOnly&&>, "");
static_assert(!cuda::std::assignable_from<void, int>, "");
static_assert(!cuda::std::assignable_from<void, void>, "");

int main(int, char**)
{
  return 0;
}
