/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

// modulus

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator%(const with_device_op&, const with_device_op&)
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
  const cuda::std::modulus<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  typedef cuda::std::modulus<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(f(36, 8) == 4);

  typedef cuda::std::modulus<> F2;
  const F2 f2 = F2();
  assert(f2(36, 8) == 4);
  assert(f2(36L, 8) == 4);
  assert(f2(36, 8L) == 4);
  constexpr int foo = cuda::std::modulus<int>()(3, 2);
  static_assert(foo == 1, "");

  constexpr int bar = cuda::std::modulus<>()(3L, 2);
  static_assert(bar == 1, "");

  return 0;
}
