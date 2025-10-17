/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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
// template <class R, class Visitor, class... Variants>
// constexpr R visit(Visitor&& vis, Variants&&... vars);

#include <uscl/std/cassert>
// #include <uscl/std/memory>
// #include <uscl/std/string>
#include <uscl/std/type_traits>
#include <uscl/std/utility>
#include <uscl/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct almost_string
{
  const char* ptr;

  __host__ __device__ almost_string(const char* ptr)
      : ptr(ptr)
  {}

  __host__ __device__ friend bool operator==(const almost_string& lhs, const almost_string& rhs)
  {
    return lhs.ptr == rhs.ptr;
  }
};

template <typename ReturnType>
__host__ __device__ void test_return_type()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;
  unused(cobj);
  { // test call operator forwarding - no variant
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(obj)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cobj)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj))), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj))), ReturnType>, "");
  }
  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    V v(42);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(obj, v)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cobj, v)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj), v)), ReturnType>,
                  "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj), v)), ReturnType>,
                  "");
    unused(v);
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(obj, v)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cobj, v)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj), v)), ReturnType>,
                  "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj), v)), ReturnType>,
                  "");
    unused(v);
  }
  { // test call operator forwarding - multi variant, multi arg
    using V  = cuda::std::variant<int, long, double>;
    using V2 = cuda::std::variant<int*, almost_string>;
    V v(42l);
    V2 v2("hello");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(obj, v, v2)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cobj, v, v2)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj), v, v2)), ReturnType>,
                  "");
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj), v, v2)), ReturnType>, "");
    unused(v, v2);
  }
  {
    using V = cuda::std::variant<int, long, double, almost_string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(obj, v1, v2, v3, v4)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cobj, v1, v2, v3, v4)), ReturnType>, "");
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj), v1, v2, v3, v4)), ReturnType>,
      "");
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj), v1, v2, v3, v4)), ReturnType>,
      "");
    unused(v1, v2, v3, v4);
  }
  {
    using V = cuda::std::variant<int, long, double, int*, almost_string>;
    V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(obj, v1, v2, v3, v4)), ReturnType>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cobj, v1, v2, v3, v4)), ReturnType>, "");
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(obj), v1, v2, v3, v4)), ReturnType>,
      "");
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::visit<ReturnType>(cuda::std::move(cobj), v1, v2, v3, v4)), ReturnType>,
      "");
    unused(v1, v2, v3, v4);
  }
}

int main(int, char**)
{
  test_return_type<void>();
  test_return_type<int>();

  return 0;
}
