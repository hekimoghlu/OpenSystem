/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

// constexpr T* optional<T>::operator->();

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  __host__ __device__ constexpr int test() noexcept
  {
    return 3;
  }
};

struct Y
{
  __host__ __device__ constexpr int test()
  {
    return 5;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::optional<X> opt{};
    unused(opt);
    static_assert(cuda::std::is_same_v<decltype(opt.operator->()), X*>);
    // static_assert(!noexcept(opt.operator->()));
    // FIXME: This assertion fails with GCC because it can see that
    // (A) operator->() is constexpr, and
    // (B) there is no path through the function that throws.
    // It's arguable if this is the correct behavior for the noexcept
    // operator.
    // Regardless this function should still be noexcept(false) because
    // it has a narrow contract.

    cuda::std::optional<X&> optref;
    unused(optref);
    static_assert(cuda::std::is_same_v<decltype(optref.operator->()), X*>);
    static_assert(noexcept(optref.operator->()));
  }

  {
    optional<X> opt(X{});
    assert(opt->test() == 3);
  }

  {
    X val{};
    optional<X&> opt(val);
    assert(opt->test() == 3);
    assert(cuda::std::addressof(val) == opt.operator->());
  }

  {
    optional<Y> opt(Y{});
    assert(opt->test() == 5);
  }

  {
    Y val{};
    optional<Y&> opt(val);
    assert(opt->test() == 5);
    assert(cuda::std::addressof(val) == opt.operator->());
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
