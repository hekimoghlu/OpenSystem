/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <uscl/std/variant>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

struct empty_visitor
{
  template <class T>
  __host__ __device__ constexpr void operator()(T) const noexcept
  {}
};

struct holder_visitor
{
  template <class T>
  __host__ __device__ constexpr Holder<Incomplete>* operator()(T) const noexcept
  {
    return nullptr;
  }
};

__host__ __device__ constexpr bool test(bool do_it)
{
  if (do_it)
  {
    cuda::std::variant<Holder<Incomplete>*, int> v = nullptr;
    cuda::std::visit(empty_visitor{}, v);
    cuda::std::visit(holder_visitor{}, v);
    cuda::std::visit<void>(empty_visitor{}, v);
    cuda::std::visit<void*>(holder_visitor{}, v);
  }
  return true;
}

int main(int, char**)
{
  test(true);
  static_assert(test(true), "");
  return 0;
}
