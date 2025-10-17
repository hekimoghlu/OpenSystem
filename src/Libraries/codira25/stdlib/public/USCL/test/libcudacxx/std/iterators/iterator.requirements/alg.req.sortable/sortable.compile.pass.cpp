/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<class I, class R = ranges::less, class P = identity>
//   concept sortable = see below;                            // since C++20

#include <uscl/std/functional>
#include <uscl/std/iterator>

using CompInt     = bool (*)(int, int);
using CompDefault = cuda::std::ranges::less;

using AllConstraintsSatisfied = int*;
static_assert(cuda::std::permutable<AllConstraintsSatisfied>, "");
static_assert(cuda::std::indirect_strict_weak_order<CompDefault, AllConstraintsSatisfied>, "");
static_assert(cuda::std::sortable<AllConstraintsSatisfied>, "");
static_assert(cuda::std::indirect_strict_weak_order<CompInt, AllConstraintsSatisfied>, "");
static_assert(cuda::std::sortable<AllConstraintsSatisfied, CompInt>, "");

struct Foo
{};
using Proj = int (*)(Foo);
static_assert(cuda::std::permutable<Foo*>, "");
static_assert(!cuda::std::indirect_strict_weak_order<CompDefault, Foo*>, "");
static_assert(cuda::std::indirect_strict_weak_order<CompDefault, cuda::std::projected<Foo*, Proj>>, "");
static_assert(!cuda::std::sortable<Foo*, CompDefault>, "");
static_assert(cuda::std::sortable<Foo*, CompDefault, Proj>, "");
static_assert(!cuda::std::indirect_strict_weak_order<CompInt, Foo*>, "");
static_assert(cuda::std::indirect_strict_weak_order<CompInt, cuda::std::projected<Foo*, Proj>>, "");
static_assert(!cuda::std::sortable<Foo*, CompInt>, "");
static_assert(cuda::std::sortable<Foo*, CompInt, Proj>, "");

using NotPermutable = const int*;
static_assert(!cuda::std::permutable<NotPermutable>, "");
static_assert(cuda::std::indirect_strict_weak_order<CompInt, NotPermutable>, "");
static_assert(!cuda::std::sortable<NotPermutable, CompInt>, "");

struct Empty
{};
using NoIndirectStrictWeakOrder = Empty*;
static_assert(cuda::std::permutable<NoIndirectStrictWeakOrder>, "");
static_assert(!cuda::std::indirect_strict_weak_order<CompInt, NoIndirectStrictWeakOrder>, "");
static_assert(!cuda::std::sortable<NoIndirectStrictWeakOrder, CompInt>, "");

int main(int, char**)
{
  return 0;
}
