/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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

// template <size_t I, class... Types>
//  constexpr add_pointer_t<variant_alternative_t<I, variant<Types...>>>
//   get_if(variant<Types...>* v) noexcept;
// template <size_t I, class... Types>
//  constexpr add_pointer_t<const variant_alternative_t<I, variant<Types...>>>
//   get_if(const variant<Types...>* v) noexcept;

#include <uscl/std/cassert>
// #include <uscl/std/memory>
#include <uscl/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

__host__ __device__ void test_const_get_if()
{
  {
    using V              = cuda::std::variant<int>;
    constexpr const V* v = nullptr;
    static_assert(cuda::std::get_if<0>(v) == nullptr, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42);
    static_assert(noexcept(cuda::std::get_if<0>(&v)));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), const int*>);
#if defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(*cuda::std::get_if<0>(&v) == 42, "");
#endif // _CCCL_BUILTIN_ADDRESSOF
    static_assert(cuda::std::get_if<1>(&v) == nullptr, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<1>(&v)), const long*>);
#if defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(*cuda::std::get_if<1>(&v) == 42, "");
#endif // _CCCL_BUILTIN_ADDRESSOF
    static_assert(cuda::std::get_if<0>(&v) == nullptr, "");
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    const V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), const int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
#endif
}

__host__ __device__ void test_get_if()
{
  {
    using V = cuda::std::variant<int>;
    V* v    = nullptr;
    assert(cuda::std::get_if<0>(v) == nullptr);
  }
  {
    using V = cuda::std::variant<int, long>;
    V v(42);
    static_assert(noexcept(cuda::std::get_if<0>(&v)));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), int*>);
    assert(*cuda::std::get_if<0>(&v) == 42);
    assert(cuda::std::get_if<1>(&v) == nullptr);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<1>(&v)), const long*>);
    assert(*cuda::std::get_if<1>(&v) == 42);
    assert(cuda::std::get_if<0>(&v) == nullptr);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x   = 42;
    V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), const int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get_if<0>(&v)), const int*>);
    assert(cuda::std::get_if<0>(&v) == &x);
  }
#endif
}

int main(int, char**)
{
  test_const_get_if();
  test_get_if();

  return 0;
}
