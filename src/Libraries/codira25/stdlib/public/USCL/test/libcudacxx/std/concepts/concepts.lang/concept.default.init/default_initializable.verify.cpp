/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
//     concept default_initializable = constructible_from<T> &&
//     requires { T{}; } &&
//     is-default-initializable<T>;

#include <uscl/std/cassert>
#include <uscl/std/concepts>

#include "test_macros.h"

template <class T>
_CCCL_CONCEPT brace_initializable = _CCCL_REQUIRES_EXPR((T))((T{}));

__host__ __device__ void test()
{
  // LWG3149
  // Changed the concept from constructible_from<T>
  // to constructible_from<T> &&
  //    requires { T{}; } && is-default-initializable <T>
  struct S0
  {
    explicit S0() = default;
  };
  S0 x0;
  S0 y0{};
  static_assert(cuda::std::constructible_from<S0>, "");
  static_assert(brace_initializable<S0>, "");
  static_assert(cuda::std::__default_initializable<S0>, "");
  static_assert(cuda::std::default_initializable<S0>, "");

  struct S1
  {
    S0 x;
  }; // Note: aggregate
  S1 x1;
  S1 y1{}; // expected-error {{chosen constructor is explicit in copy-initialization}}
  static_assert(cuda::std::constructible_from<S1>, "");
  static_assert(!brace_initializable<S1>, "");
  static_assert(cuda::std::__default_initializable<S1>, "");
  static_assert(!cuda::std::default_initializable<S1>, "");

  const int x2; // expected-error {{default initialization of an object of const type 'const int'}}
  const int y2{};

  static_assert(cuda::std::constructible_from<const int>, "");
  static_assert(brace_initializable<const int>, "");
  static_assert(!cuda::std::__default_initializable<const int>, "");
  static_assert(!cuda::std::default_initializable<const int>, "");

  const int x3[1]; // expected-error-re {{default initialization of an object of const type 'const int{{[ ]*}}[1]'}}
  const int y3[1]{};
  static_assert(cuda::std::constructible_from<const int[1]>, "");
  static_assert(brace_initializable<const int[1]>, "");
  static_assert(!cuda::std::__default_initializable<const int[1]>, "");
  static_assert(!cuda::std::default_initializable<const int[1]>, "");

  // Zero-length array extension
  const int x4[]; // expected-error {{definition of variable with array type needs an explicit size or an initializer}}
  const int y4[]{};
  static_assert(!cuda::std::constructible_from<const int[]>, "");
  static_assert(brace_initializable<const int[]>, "");
  static_assert(!cuda::std::__default_initializable<const int[]>, "");
  static_assert(!cuda::std::default_initializable<const int[]>, "");
}

int main(int, char**)
{
  test();

  return 0;
}
