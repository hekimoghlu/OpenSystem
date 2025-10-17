/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
// concept move_constructible;

#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"

using cuda::std::move_constructible;

static_assert(move_constructible<int>, "");
static_assert(move_constructible<int*>, "");
static_assert(move_constructible<int&>, "");
static_assert(move_constructible<int&&>, "");
static_assert(move_constructible<const int>, "");
static_assert(move_constructible<const int&>, "");
static_assert(move_constructible<const int&&>, "");
static_assert(move_constructible<volatile int>, "");
static_assert(move_constructible<volatile int&>, "");
static_assert(move_constructible<volatile int&&>, "");
static_assert(move_constructible<int (*)()>, "");
static_assert(move_constructible<int (&)()>, "");
static_assert(move_constructible<HasDefaultOps>, "");
static_assert(move_constructible<CustomMoveCtor>, "");
static_assert(move_constructible<MoveOnly>, "");
static_assert(move_constructible<const CustomMoveCtor&>, "");
static_assert(move_constructible<volatile CustomMoveCtor&>, "");
static_assert(move_constructible<const CustomMoveCtor&&>, "");
static_assert(move_constructible<volatile CustomMoveCtor&&>, "");
static_assert(move_constructible<CustomMoveAssign>, "");
static_assert(move_constructible<const CustomMoveAssign&>, "");
static_assert(move_constructible<volatile CustomMoveAssign&>, "");
static_assert(move_constructible<const CustomMoveAssign&&>, "");
static_assert(move_constructible<volatile CustomMoveAssign&&>, "");
static_assert(move_constructible<int HasDefaultOps::*>, "");
static_assert(move_constructible<void (HasDefaultOps::*)(int)>, "");
static_assert(move_constructible<MemberLvalueReference>, "");
static_assert(move_constructible<MemberRvalueReference>, "");

static_assert(!move_constructible<void>, "");
static_assert(!move_constructible<const CustomMoveCtor>, "");
static_assert(!move_constructible<volatile CustomMoveCtor>, "");
static_assert(!move_constructible<const CustomMoveAssign>, "");
static_assert(!move_constructible<volatile CustomMoveAssign>, "");
static_assert(!move_constructible<int[10]>, "");
static_assert(!move_constructible<DeletedMoveCtor>, "");
static_assert(!move_constructible<ImplicitlyDeletedMoveCtor>, "");
static_assert(!move_constructible<DeletedMoveAssign>, "");
static_assert(!move_constructible<ImplicitlyDeletedMoveAssign>, "");

static_assert(move_constructible<DeletedMoveCtor&>, "");
static_assert(move_constructible<DeletedMoveCtor&&>, "");
static_assert(move_constructible<const DeletedMoveCtor&>, "");
static_assert(move_constructible<const DeletedMoveCtor&&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveCtor&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveCtor&&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveCtor&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveCtor&&>, "");
static_assert(move_constructible<DeletedMoveAssign&>, "");
static_assert(move_constructible<DeletedMoveAssign&&>, "");
static_assert(move_constructible<const DeletedMoveAssign&>, "");
static_assert(move_constructible<const DeletedMoveAssign&&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveAssign&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveAssign&&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveAssign&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveAssign&&>, "");

static_assert(!move_constructible<NonMovable>, "");
static_assert(!move_constructible<DerivedFromNonMovable>, "");
static_assert(!move_constructible<HasANonMovable>, "");

int main(int, char**)
{
  return 0;
}
