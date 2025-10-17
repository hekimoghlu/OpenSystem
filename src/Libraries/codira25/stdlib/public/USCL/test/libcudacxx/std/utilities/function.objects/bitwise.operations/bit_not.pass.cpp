/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

// <functional>

// bit_not

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator~(const with_device_op&)
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
  const cuda::std::bit_not<with_device_op> f;
  assert(f({}));
}

int main(int, char**)
{
  typedef cuda::std::bit_not<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<F::argument_type, int>::value), "");
  static_assert((cuda::std::is_same<F::result_type, int>::value), "");
#endif // TEST_STD_VER <= 2017
  assert((f(0xEA95) & 0xFFFF) == 0x156A);
  assert((f(0x58D3) & 0xFFFF) == 0xA72C);
  assert((f(0) & 0xFFFF) == 0xFFFF);
  assert((f(0xFFFF) & 0xFFFF) == 0);

  typedef cuda::std::bit_not<> F2;
  const F2 f2 = F2();
  assert((f2(0xEA95) & 0xFFFF) == 0x156A);
  assert((f2(0xEA95L) & 0xFFFF) == 0x156A);
  assert((f2(0x58D3) & 0xFFFF) == 0xA72C);
  assert((f2(0x58D3L) & 0xFFFF) == 0xA72C);
  assert((f2(0) & 0xFFFF) == 0xFFFF);
  assert((f2(0L) & 0xFFFF) == 0xFFFF);
  assert((f2(0xFFFF) & 0xFFFF) == 0);
  assert((f2(0xFFFFL) & 0xFFFF) == 0);

  constexpr int foo = cuda::std::bit_not<int>()(0xEA95) & 0xFFFF;
  static_assert(foo == 0x156A, "");

  constexpr int bar = cuda::std::bit_not<>()(0xEA95) & 0xFFFF;
  static_assert(bar == 0x156A, "");

  return 0;
}
