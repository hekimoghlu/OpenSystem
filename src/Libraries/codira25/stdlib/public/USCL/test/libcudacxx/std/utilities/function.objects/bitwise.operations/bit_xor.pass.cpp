/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

// <cuda/std/functional>

// bit_xor

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator^(const with_device_op&, const with_device_op&)
  {
    return {};
  }
  __device__ constexpr operator bool() const
  {
    return true;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::bit_xor<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  {
    typedef cuda::std::bit_xor<int> F;
    const F f = F();
#if TEST_STD_VER <= 2017
    static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
    static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
    static_assert((cuda::std::is_same<int, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
    assert(f(0xEA95, 0xEA95) == 0);
    assert(f(0xEA95, 0x58D3) == 0xB246);
    assert(f(0x58D3, 0xEA95) == 0xB246);
    assert(f(0x58D3, 0) == 0x58D3);
    assert(f(0xFFFF, 0x58D3) == 0xA72C);
  }

  {
    typedef cuda::std::bit_xor<> F2;
    const F2 f = F2();
    assert(f(0xEA95, 0xEA95) == 0);
    assert(f(0xEA95L, 0xEA95) == 0);
    assert(f(0xEA95, 0xEA95L) == 0);

    assert(f(0xEA95, 0x58D3) == 0xB246);
    assert(f(0xEA95L, 0x58D3) == 0xB246);
    assert(f(0xEA95, 0x58D3L) == 0xB246);

    assert(f(0x58D3, 0xEA95) == 0xB246);
    assert(f(0x58D3L, 0xEA95) == 0xB246);
    assert(f(0x58D3, 0xEA95L) == 0xB246);

    assert(f(0x58D3, 0) == 0x58D3);
    assert(f(0x58D3L, 0) == 0x58D3);
    assert(f(0x58D3, 0L) == 0x58D3);

    assert(f(0xFFFF, 0x58D3) == 0xA72C);
    assert(f(0xFFFFL, 0x58D3) == 0xA72C);
    assert(f(0xFFFF, 0x58D3L) == 0xA72C);
    constexpr int foo = cuda::std::bit_xor<int>()(0x58D3, 0xEA95);
    static_assert(foo == 0xB246, "");

    constexpr int bar = cuda::std::bit_xor<>()(0x58D3L, 0xEA95);
    static_assert(bar == 0xB246, "");
  }

  return 0;
}
