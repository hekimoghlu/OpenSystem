/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

#include <uscl/std/cassert>
// #include <uscl/std/memory>
// #include <uscl/std/string>
#include <uscl/std/type_traits>
#include <uscl/std/utility>
#include <uscl/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

__host__ __device__ void test_constexpr()
{
  constexpr ReturnFirst obj{};
  constexpr ReturnArity aobj{};
  {
    using V = cuda::std::variant<int>;
    constexpr V v(42);
    static_assert(cuda::std::visit(obj, v) == 42, "");
  }
  {
    using V = cuda::std::variant<short, long, char>;
    constexpr V v(42l);
    static_assert(cuda::std::visit(obj, v) == 42, "");
  }
  {
    using V1 = cuda::std::variant<int>;
    using V2 = cuda::std::variant<int, char*, long long>;
    using V3 = cuda::std::variant<bool, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert(cuda::std::visit(aobj, v1, v2, v3) == 3, "");
  }
  {
    using V1 = cuda::std::variant<int>;
    using V2 = cuda::std::variant<int, char*, long long>;
    using V3 = cuda::std::variant<void*, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert(cuda::std::visit(aobj, v1, v2, v3) == 3, "");
  }
  {
    using V = cuda::std::variant<int, long, double, int*>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert(cuda::std::visit(aobj, v1, v2, v3, v4) == 4, "");
  }
  {
    using V = cuda::std::variant<int, long, double, long long, int*>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert(cuda::std::visit(aobj, v1, v2, v3, v4) == 4, "");
  }
}

int main(int, char**)
{
  test_constexpr();

  return 0;
}
