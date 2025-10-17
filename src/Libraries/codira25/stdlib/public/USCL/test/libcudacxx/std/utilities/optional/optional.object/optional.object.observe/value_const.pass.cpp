/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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

// constexpr const T& optional<T>::value() const &;

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::in_place;
using cuda::std::in_place_t;
using cuda::std::optional;
#if TEST_HAS_EXCEPTIONS()
using cuda::std::bad_optional_access;
#endif

struct X
{
  X()         = default;
  X(const X&) = delete;
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

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  const optional<X> opt{};
  try
  {
    (void) opt.value();
    assert(false);
  }
  catch (const bad_optional_access&)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ constexpr bool test()
{
  {
    const optional<X> opt{};
    unused(opt);
    static_assert(!noexcept(opt.value()));
    static_assert(cuda::std::is_same_v<decltype(opt.value()), const X&>);

    const optional<X&> optref;
    unused(optref);
    static_assert(noexcept(optref.value()));
    static_assert(cuda::std::is_same_v<decltype(optref.value()), X&>);
  }

  {
    const optional<X> opt{cuda::std::in_place};
    assert(opt.value().test() == 3);
  }

  {
    X val{};
    const optional<X&> opt{val};
    assert(opt.value().test() == 4);
    assert(cuda::std::addressof(val) == cuda::std::addressof(opt.value()));
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
