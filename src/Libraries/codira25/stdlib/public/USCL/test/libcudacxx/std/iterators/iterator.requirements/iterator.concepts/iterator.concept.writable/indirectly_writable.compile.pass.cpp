/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

// template<class In>
// concept indirectly_writable;

#include <uscl/std/concepts>
#include <uscl/std/iterator>

#include "read_write.h"

template <class Out, class T>
__host__ __device__ constexpr bool check_indirectly_writable()
{
  constexpr bool result = cuda::std::indirectly_writable<Out, T>;
  static_assert(cuda::std::indirectly_writable<Out const, T> == result, "");
  return result;
}

static_assert(check_indirectly_writable<value_type_indirection, int>(), "");
static_assert(check_indirectly_writable<value_type_indirection, double>(), "");
static_assert(!check_indirectly_writable<value_type_indirection, double*>(), "");

static_assert(!check_indirectly_writable<read_only_indirection, int>(), "");
static_assert(!check_indirectly_writable<proxy_indirection, int>(), "");

static_assert(!check_indirectly_writable<int, int>(), "");
static_assert(!check_indirectly_writable<missing_dereference, missing_dereference::value_type>(), "");

static_assert(!check_indirectly_writable<void*, int>(), "");
static_assert(!check_indirectly_writable<void const*, int>(), "");
static_assert(!check_indirectly_writable<void volatile*, int>(), "");
static_assert(!check_indirectly_writable<void const volatile*, int>(), "");
static_assert(!check_indirectly_writable<void*, double>(), "");
static_assert(check_indirectly_writable<void**, int*>(), "");
static_assert(!check_indirectly_writable<void**, int>(), "");

static_assert(check_indirectly_writable<int*, int>(), "");
static_assert(!check_indirectly_writable<int const*, int>(), "");
static_assert(check_indirectly_writable<int volatile*, int>(), "");
static_assert(!check_indirectly_writable<int const volatile*, int>(), "");
static_assert(check_indirectly_writable<int*, double>(), "");
static_assert(check_indirectly_writable<int**, int*>(), "");
static_assert(!check_indirectly_writable<int**, int>(), "");

int main(int, char**)
{
  return 0;
}
