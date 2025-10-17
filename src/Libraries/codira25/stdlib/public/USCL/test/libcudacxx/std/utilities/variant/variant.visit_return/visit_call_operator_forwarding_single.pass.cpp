/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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

template <typename ReturnType>
__host__ __device__ void test_call_operator_forwarding()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;
  { // test call operator forwarding - no variant
    cuda::std::visit<ReturnType>(obj);
    assert(Fn::check_call<>(CT_NonConst | CT_LValue));
    cuda::std::visit<ReturnType>(cobj);
    assert(Fn::check_call<>(CT_Const | CT_LValue));
    cuda::std::visit<ReturnType>(cuda::std::move(obj));
    assert(Fn::check_call<>(CT_NonConst | CT_RValue));
    cuda::std::visit<ReturnType>(cuda::std::move(cobj));
    assert(Fn::check_call<>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    V v(42);
    cuda::std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
    cuda::std::visit<ReturnType>(cobj, v);
    assert(Fn::check_call<int&>(CT_Const | CT_LValue));
    cuda::std::visit<ReturnType>(cuda::std::move(obj), v);
    assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
    cuda::std::visit<ReturnType>(cuda::std::move(cobj), v);
    assert(Fn::check_call<int&>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    V v(42l);
    cuda::std::visit<ReturnType>(obj, v);
    assert(Fn::check_call<long&>(CT_NonConst | CT_LValue));
    cuda::std::visit<ReturnType>(cobj, v);
    assert(Fn::check_call<long&>(CT_Const | CT_LValue));
    cuda::std::visit<ReturnType>(cuda::std::move(obj), v);
    assert(Fn::check_call<long&>(CT_NonConst | CT_RValue));
    cuda::std::visit<ReturnType>(cuda::std::move(cobj), v);
    assert(Fn::check_call<long&>(CT_Const | CT_RValue));
  }
}

int main(int, char**)
{
  test_call_operator_forwarding<void>();
  test_call_operator_forwarding<int>();

  return 0;
}
