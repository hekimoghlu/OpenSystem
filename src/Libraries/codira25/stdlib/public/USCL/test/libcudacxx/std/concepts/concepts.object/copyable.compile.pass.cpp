/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
// concept copyable = see below;

#include "type_classification/copyable.h"

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::copyable;

static_assert(copyable<int>, "");
static_assert(copyable<int volatile>, "");
static_assert(copyable<int*>, "");
static_assert(copyable<int const*>, "");
static_assert(copyable<int volatile*>, "");
static_assert(copyable<int volatile const*>, "");
static_assert(copyable<int (*)()>, "");

struct S
{};
static_assert(copyable<S>, "");
static_assert(copyable<int S::*>, "");
static_assert(copyable<int (S::*)()>, "");
static_assert(copyable<int (S::*)() noexcept>, "");
static_assert(copyable<int (S::*)() &>, "");
static_assert(copyable<int (S::*)() & noexcept>, "");
static_assert(copyable<int (S::*)() &&>, "");
static_assert(copyable < int (S::*)() && noexcept >, "");
static_assert(copyable<int (S::*)() const>, "");
static_assert(copyable<int (S::*)() const noexcept>, "");
static_assert(copyable<int (S::*)() const&>, "");
static_assert(copyable<int (S::*)() const & noexcept>, "");
static_assert(copyable<int (S::*)() const&&>, "");
static_assert(copyable < int (S::*)() const&& noexcept >, "");
static_assert(copyable<int (S::*)() volatile>, "");
static_assert(copyable<int (S::*)() volatile noexcept>, "");
static_assert(copyable<int (S::*)() volatile&>, "");
static_assert(copyable<int (S::*)() volatile & noexcept>, "");
static_assert(copyable<int (S::*)() volatile&&>, "");
static_assert(copyable < int (S::*)() volatile && noexcept >, "");
static_assert(copyable<int (S::*)() const volatile>, "");
static_assert(copyable<int (S::*)() const volatile noexcept>, "");
static_assert(copyable<int (S::*)() const volatile&>, "");
static_assert(copyable<int (S::*)() const volatile & noexcept>, "");
static_assert(copyable<int (S::*)() const volatile&&>, "");
static_assert(copyable < int (S::*)() const volatile&& noexcept >, "");

static_assert(copyable<has_volatile_member>, "");
static_assert(copyable<has_array_member>, "");

// Not objects
static_assert(!copyable<void>, "");
static_assert(!copyable<int&>, "");
static_assert(!copyable<int const&>, "");
static_assert(!copyable<int volatile&>, "");
static_assert(!copyable<int const volatile&>, "");
static_assert(!copyable<int&&>, "");
static_assert(!copyable<int const&&>, "");
static_assert(!copyable<int volatile&&>, "");
static_assert(!copyable<int const volatile&&>, "");
static_assert(!copyable<int()>, "");
static_assert(!copyable<int (&)()>, "");
static_assert(!copyable<int[5]>, "");

// Not assignable
static_assert(!copyable<int const>, "");
static_assert(!copyable<int const volatile>, "");
static_assert(copyable<const_copy_assignment const>, "");
static_assert(!copyable<volatile_copy_assignment volatile>, "");
static_assert(copyable<cv_copy_assignment const volatile>, "");

static_assert(!copyable<no_copy_constructor>, "");
static_assert(!copyable<no_copy_assignment>, "");

#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions
                                                // of SMF
static_assert(cuda::std::is_copy_assignable_v<no_copy_assignment_mutable>, "");
static_assert(!copyable<no_copy_assignment_mutable>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(!copyable<derived_from_noncopyable>, "");
static_assert(!copyable<has_noncopyable>, "");
static_assert(!copyable<has_const_member>, "");
static_assert(!copyable<has_cv_member>, "");
static_assert(!copyable<has_lvalue_reference_member>, "");
static_assert(!copyable<has_rvalue_reference_member>, "");
static_assert(!copyable<has_function_ref_member>, "");

static_assert(
  !cuda::std::assignable_from<deleted_assignment_from_const_rvalue&, deleted_assignment_from_const_rvalue const>, "");
static_assert(!copyable<deleted_assignment_from_const_rvalue>, "");

int main(int, char**)
{
  return 0;
}
