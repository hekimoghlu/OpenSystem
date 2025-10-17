/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
// UNSUPPORTED: gcc-6

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
__host__ __device__ void test_argument_forwarding()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const auto Val = CT_LValue | CT_NonConst;
  { // single argument - value type
    using V = cuda::std::variant<int>;
    V v(42);
    const V& cv = v;
    cuda::std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int&>(Val));
    cuda::std::visit<ReturnType>(obj, cv);
    assert(Fn::check_call<const int&>(Val));
    cuda::std::visit<ReturnType>(obj, cuda::std::move(v));
    assert(Fn::check_call<int&&>(Val));
    cuda::std::visit<ReturnType>(obj, cuda::std::move(cv));
    assert(Fn::check_call<const int&&>(Val));
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  { // single argument - lvalue reference
    using V = cuda::std::variant<int&>;
    int x   = 42;
    V v(x);
    const V& cv = v;
    cuda::std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int&>(Val));
    cuda::std::visit<ReturnType>(obj, cv);
    assert(Fn::check_call<int&>(Val));
    cuda::std::visit<ReturnType>(obj, cuda::std::move(v));
    assert(Fn::check_call<int&>(Val));
    cuda::std::visit<ReturnType>(obj, cuda::std::move(cv));
    assert(Fn::check_call<int&>(Val));
  }
  { // single argument - rvalue reference
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    const V& cv = v;
    cuda::std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int&>(Val));
    cuda::std::visit<ReturnType>(obj, cv);
    assert(Fn::check_call<int&>(Val));
    cuda::std::visit<ReturnType>(obj, cuda::std::move(v));
    assert(Fn::check_call<int&&>(Val));
    cuda::std::visit<ReturnType>(obj, cuda::std::move(cv));
    assert(Fn::check_call<int&&>(Val));
  }
#endif
  { // multi argument - multi variant
    using V = cuda::std::variant<int, almost_string, long>;
    V v1(42), v2("hello"), v3(43l);
    cuda::std::visit<ReturnType>(obj, v1, v2, v3);
    assert((Fn::check_call<int&, almost_string&, long&>(Val)));
    cuda::std::visit<ReturnType>(obj, cuda::std::as_const(v1), cuda::std::as_const(v2), cuda::std::move(v3));
    assert((Fn::check_call<const int&, const almost_string&, long&&>(Val)));
  }
  {
    using V = cuda::std::variant<int, long, double, almost_string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    cuda::std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long&, almost_string&, int&, double&>(Val)));
    cuda::std::visit<ReturnType>(
      obj, cuda::std::as_const(v1), cuda::std::as_const(v2), cuda::std::move(v3), cuda::std::move(v4));
    assert((Fn::check_call<const long&, const almost_string&, int&&, double&&>(Val)));
  }
  {
    using V = cuda::std::variant<int, long, double, int*, almost_string>;
    V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    cuda::std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long&, almost_string&, int*&, double&>(Val)));
    cuda::std::visit<ReturnType>(
      obj, cuda::std::as_const(v1), cuda::std::as_const(v2), cuda::std::move(v3), cuda::std::move(v4));
    assert((Fn::check_call<const long&, const almost_string&, int*&&, double&&>(Val)));
  }
}

int main(int, char**)
{
  test_argument_forwarding<void>();
  test_argument_forwarding<int>();

  return 0;
}
