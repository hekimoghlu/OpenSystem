/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
// concept movable = see below;

#include "type_classification/movable.h"

#include <uscl/std/concepts>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"

using cuda::std::movable;

// Movable types
static_assert(movable<int>, "");
static_assert(movable<int volatile>, "");
static_assert(movable<int*>, "");
static_assert(movable<int const*>, "");
static_assert(movable<int volatile*>, "");
static_assert(movable<int const volatile*>, "");
static_assert(movable<int (*)()>, "");

struct S
{};
static_assert(movable<S>, "");
static_assert(movable<int S::*>, "");
static_assert(movable<int (S::*)()>, "");
static_assert(movable<int (S::*)() noexcept>, "");
static_assert(movable<int (S::*)() &>, "");
static_assert(movable<int (S::*)() & noexcept>, "");
static_assert(movable<int (S::*)() &&>, "");
static_assert(movable < int (S::*)() && noexcept >, "");
static_assert(movable<int (S::*)() const>, "");
static_assert(movable<int (S::*)() const noexcept>, "");
static_assert(movable<int (S::*)() const&>, "");
static_assert(movable<int (S::*)() const & noexcept>, "");
static_assert(movable<int (S::*)() const&&>, "");
static_assert(movable < int (S::*)() const&& noexcept >, "");
static_assert(movable<int (S::*)() volatile>, "");
static_assert(movable<int (S::*)() volatile noexcept>, "");
static_assert(movable<int (S::*)() volatile&>, "");
static_assert(movable<int (S::*)() volatile & noexcept>, "");
static_assert(movable<int (S::*)() volatile&&>, "");
static_assert(movable < int (S::*)() volatile && noexcept >, "");
static_assert(movable<int (S::*)() const volatile>, "");
static_assert(movable<int (S::*)() const volatile noexcept>, "");
static_assert(movable<int (S::*)() const volatile&>, "");
static_assert(movable<int (S::*)() const volatile & noexcept>, "");
static_assert(movable<int (S::*)() const volatile&&>, "");
static_assert(movable < int (S::*)() const volatile&& noexcept >, "");

static_assert(movable<has_volatile_member>, "");
static_assert(movable<has_array_member>, "");

// Not objects
static_assert(!movable<int&>, "");
static_assert(!movable<int const&>, "");
static_assert(!movable<int volatile&>, "");
static_assert(!movable<int const volatile&>, "");
static_assert(!movable<int&&>, "");
static_assert(!movable<int const&&>, "");
static_assert(!movable<int volatile&&>, "");
static_assert(!movable<int const volatile&&>, "");
static_assert(!movable<int()>, "");
static_assert(!movable<int (&)()>, "");
static_assert(!movable<int[5]>, "");

// Core non-move assignable.
static_assert(!movable<int const>, "");
static_assert(!movable<int const volatile>, "");

static_assert(!movable<DeletedMoveCtor>, "");
static_assert(!movable<ImplicitlyDeletedMoveCtor>, "");
static_assert(!movable<DeletedMoveAssign>, "");
static_assert(!movable<ImplicitlyDeletedMoveAssign>, "");
static_assert(!movable<NonMovable>, "");
static_assert(!movable<DerivedFromNonMovable>, "");
static_assert(!movable<HasANonMovable>, "");

static_assert(movable<cpp03_friendly>, "");
static_assert(movable<const_move_ctor>, "");
static_assert(movable<volatile_move_ctor>, "");
static_assert(movable<cv_move_ctor>, "");
static_assert(movable<multi_param_move_ctor>, "");
static_assert(!movable<not_quite_multi_param_move_ctor>, "");

#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions
                                                // of SMF
static_assert(!cuda::std::assignable_from<copy_with_mutable_parameter&, copy_with_mutable_parameter>, "");
static_assert(!movable<copy_with_mutable_parameter>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017

static_assert(!movable<const_move_assignment>, "");
static_assert(movable<volatile_move_assignment>, "");
static_assert(!movable<cv_move_assignment>, "");

static_assert(!movable<const_move_assign_and_traditional_move_assign>, "");
static_assert(!movable<volatile_move_assign_and_traditional_move_assign>, "");
static_assert(!movable<cv_move_assign_and_traditional_move_assign>, "");
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions
                                                // of SMF
static_assert(movable<const_move_assign_and_default_ops>, "");
static_assert(movable<volatile_move_assign_and_default_ops>, "");
static_assert(movable<cv_move_assign_and_default_ops>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple
       // definitions of SMF

static_assert(!movable<has_const_member>, "");
static_assert(!movable<has_cv_member>, "");
static_assert(!movable<has_lvalue_reference_member>, "");
static_assert(!movable<has_rvalue_reference_member>, "");
static_assert(!movable<has_function_ref_member>, "");

static_assert(movable<deleted_assignment_from_const_rvalue>, "");

// `move_constructible and assignable_from<T&, T>` implies `swappable<T>`,
// so there's nothing to test for the case of non-swappable.

int main(int, char**)
{
  return 0;
}
