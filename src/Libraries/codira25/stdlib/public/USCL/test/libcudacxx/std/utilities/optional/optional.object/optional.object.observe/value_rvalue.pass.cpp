/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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

// <cuda/std/optional>

// constexpr T& optional<T>::value() &&;

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;
#if TEST_HAS_EXCEPTIONS()
using cuda::std::bad_optional_access;
#endif

struct X
{
  X()                    = default;
  X(const X&)            = delete;
  X& operator=(const X&) = delete;
  __host__ __device__ constexpr int test() const&
  {
    return 3;
  }
  __host__ __device__ constexpr int test() &
  {
    return 4;
  }
  __host__ __device__ constexpr int test() const&&
  {
    return 5;
  }
  __host__ __device__ constexpr int test() &&
  {
    return 6;
  }
};

struct Y
{
  __host__ __device__ constexpr int test() &&
  {
    return 7;
  }

  __host__ __device__ constexpr int test() &
  {
    return 42;
  }
};

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  optional<X> opt{};
  try
  {
    (void) cuda::std::move(opt).value();
    assert(false);
  }
  catch (const bad_optional_access&)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ constexpr bool test()
{
  {
    optional<X> opt{};
    unused(opt);
    static_assert(!noexcept(cuda::std::move(opt).value()));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(opt).value()), X&&>);

    optional<X&> optref;
    unused(optref);
    static_assert(noexcept(cuda::std::move(optref).value()));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(optref).value()), X&>);
  }

  {
    optional<X> opt{cuda::std::in_place};
    assert(cuda::std::move(opt).value().test() == 6);
  }

  {
    X val{};
    optional<X&> opt{val};
    assert(cuda::std::move(opt).value().test() == 4);
    assert(cuda::std::addressof(val) == cuda::std::addressof(cuda::std::move(opt).value()));
  }

  {
    optional<Y> opt{cuda::std::in_place};
    assert(cuda::std::move(opt).value().test() == 7);
  }

  {
    Y val{};
    optional<Y&> opt{val};
    assert(cuda::std::move(opt).value().test() == 42);
    assert(cuda::std::addressof(val) == cuda::std::addressof(cuda::std::move(opt).value()));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
