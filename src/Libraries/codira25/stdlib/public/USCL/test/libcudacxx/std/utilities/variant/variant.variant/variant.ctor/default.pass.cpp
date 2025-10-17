/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

// template <class ...Types> class variant;

// constexpr variant() noexcept(see below);

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct NonDefaultConstructible
{
  __host__ __device__ constexpr NonDefaultConstructible(int) {}
};

struct NotNoexcept
{
  __host__ __device__ NotNoexcept() noexcept(false) {}
};

#if TEST_HAS_EXCEPTIONS()
struct DefaultCtorThrows
{
  DefaultCtorThrows()
  {
    throw 42;
  }
};
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_default_ctor_sfinae()
{
  {
    using V = cuda::std::variant<cuda::std::monostate, int>;
    static_assert(cuda::std::is_default_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<NonDefaultConstructible, int>;
    static_assert(!cuda::std::is_default_constructible<V>::value, "");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&, int>;
    static_assert(!cuda::std::is_default_constructible<V>::value, "");
  }
#endif
}

__host__ __device__ void test_default_ctor_noexcept()
{
  {
    using V = cuda::std::variant<int>;
    static_assert(cuda::std::is_nothrow_default_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<NotNoexcept>;
    static_assert(!cuda::std::is_nothrow_default_constructible<V>::value, "");
  }
}

#if TEST_HAS_EXCEPTIONS()
void test_default_ctor_throws()
{
  using V = cuda::std::variant<DefaultCtorThrows, int>;
  try
  {
    V v;
    assert(false);
  }
  catch (const int& ex)
  {
    assert(ex == 42);
  }
  catch (...)
  {
    assert(false);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_default_ctor_basic()
{
  {
    cuda::std::variant<int> v;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 0);
  }
  {
    cuda::std::variant<int, long> v;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 0);
  }
  {
    cuda::std::variant<int, NonDefaultConstructible> v;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 0);
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 0, "");
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 0, "");
  }
  {
    using V = cuda::std::variant<int, NonDefaultConstructible>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 0, "");
  }
}

int main(int, char**)
{
  test_default_ctor_basic();
  test_default_ctor_sfinae();
  test_default_ctor_noexcept();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_default_ctor_throws();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
