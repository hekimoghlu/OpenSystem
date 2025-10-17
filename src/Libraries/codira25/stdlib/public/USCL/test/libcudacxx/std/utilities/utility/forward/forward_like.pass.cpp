/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

// test forward_like

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

__host__ __device__ void compile_test()
{
  struct U
  {}; // class type so const-qualification is not stripped from a prvalue
  using CU = const U;
  using T  = int;
  using CT = const T;

  U u{};
  const U& cu = u;

  unused(u, cu);

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T>(U{})), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T>(CU{})), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T>(u)), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T>(cu)), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T>(cuda::std::move(u))), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T>(cuda::std::move(cu))), CU&&>::value, "");

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT>(U{})), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT>(CU{})), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT>(u)), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT>(cu)), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT>(cuda::std::move(u))), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT>(cuda::std::move(cu))), CU&&>::value, "");

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(U{})), U&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(CU{})), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(u)), U&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(cu)), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(cuda::std::move(u))), U&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(cuda::std::move(cu))), CU&>::value, "");

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(U{})), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(CU{})), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(u)), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(cu)), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(cuda::std::move(u))), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(cuda::std::move(cu))), CU&>::value, "");

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(U{})), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(CU{})), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(u)), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(cu)), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(cuda::std::move(u))), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(cuda::std::move(cu))), CU&&>::value, "");

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(U{})), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(CU{})), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(u)), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(cu)), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(cuda::std::move(u))), CU&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(cuda::std::move(cu))), CU&&>::value, "");

  static_assert(noexcept(cuda::std::forward_like<T>(u)), "");

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<U&>(u)), U&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CU&>(cu)), CU&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<U&&>(cuda::std::move(u))), U&&>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CU&&>(cuda::std::move(cu))), CU&&>::value, "");

  struct NoCtorCopyMove
  {
    NoCtorCopyMove()                      = delete;
    NoCtorCopyMove(const NoCtorCopyMove&) = delete;
    NoCtorCopyMove(NoCtorCopyMove&&)      = delete;
  };

  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&&>(cuda::std::declval<NoCtorCopyMove>())),
                                   const NoCtorCopyMove&&>::value,
                "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<CT&>(cuda::std::declval<NoCtorCopyMove>())),
                                   const NoCtorCopyMove&>::value,
                "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&&>(cuda::std::declval<NoCtorCopyMove>())),
                                   NoCtorCopyMove&&>::value,
                "");
  static_assert(cuda::std::is_same<decltype(cuda::std::forward_like<T&>(cuda::std::declval<NoCtorCopyMove>())),
                                   NoCtorCopyMove&>::value,
                "");

  static_assert(noexcept(cuda::std::forward_like<T>(cuda::std::declval<NoCtorCopyMove>())), "");
}

__host__ __device__ constexpr bool test()
{
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<const double&>(val);
    static_assert(cuda::std::is_same<decltype(result), const int&>::value, "");
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<double&>(val);
    static_assert(cuda::std::is_same<decltype(result), int&>::value, "");
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<const double&&>(val);
    static_assert(cuda::std::is_same<decltype(result), const int&&>::value, "");
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<double&&>(val);
    static_assert(cuda::std::is_same<decltype(result), int&&>::value, "");
    assert(&result == &val);
  }
  return true;
}

int main(int, char**)
{
  compile_test();
  test();
  static_assert(test(), "");

  return 0;
}
