/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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

// indirect_result_t

#include <uscl/std/concepts>
#include <uscl/std/iterator>

static_assert(cuda::std::same_as<cuda::std::indirect_result_t<int (*)(int), int*>, int>, "");
static_assert(
  cuda::std::same_as<cuda::std::indirect_result_t<double (*)(int const&, float), int const*, float*>, double>, "");

struct S
{};
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<S (&)(int), int*>, S>, "");
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<long S::*, S*>, long&>, "");
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<S && (S::*) (), S*>, S&&>, "");
static_assert(cuda::std::same_as<cuda::std::indirect_result_t<int S::* (S::*) (int) const, S*, int*>, int S::*>, "");

template <class F, class... Is>
_CCCL_CONCEPT has_indirect_result =
  _CCCL_REQUIRES_EXPR((F, variadic Is))(typename(cuda::std::indirect_result_t<F, Is...>));

static_assert(!has_indirect_result<int (*)(int), int>, ""); // int isn't indirectly_readable
static_assert(!has_indirect_result<int, int*>, ""); // int isn't invocable

int main(int, char**)
{
  return 0;
}
