/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class Context = format_context, class... Args>
// format-arg-store<Context, Args...> make_format_args(Args&... args);

#include <uscl/std/__format_>
#include <uscl/std/cstddef>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

template <class Arg, class = void>
inline constexpr bool can_make_format_args = false;
template <class Arg>
inline constexpr bool
  can_make_format_args<Arg, cuda::std::void_t<decltype(cuda::std::make_format_args(cuda::std::declval<Arg>()))>> = true;

static_assert(can_make_format_args<int&>);
static_assert(!can_make_format_args<int>);
static_assert(!can_make_format_args<int&&>);

__host__ __device__ void test()
{
  auto i = 1;
  auto c = 'c';
  auto p = nullptr;
  auto b = false;

  [[maybe_unused]] auto store = cuda::std::make_format_args(i, p, b, c);
  static_assert(cuda::std::is_same_v<
                decltype(store),
                cuda::std::__format_arg_store<cuda::std::format_context, int, cuda::std::nullptr_t, bool, char>>);
}

int main(int, char**)
{
  test();
  return 0;
}
