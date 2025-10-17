/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

// equal_to

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <uscl/std/cassert>
#include <uscl/std/functional>
#include <uscl/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr bool operator==(const with_device_op&, const with_device_op&)
  {
    return true;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::equal_to<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  typedef cuda::std::equal_to<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<bool, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(f(36, 36));
  assert(!f(36, 6));

  typedef cuda::std::equal_to<> F2;
  const F2 f2 = F2();
  assert(f2(36, 36));
  assert(!f2(36, 6));
  assert(f2(36, 36.0));
  assert(f2(36.0, 36L));
  constexpr bool foo = cuda::std::equal_to<int>()(36, 36);
  static_assert(foo, "");

  constexpr bool bar = cuda::std::equal_to<>()(36.0, 36);
  static_assert(bar, "");

  return 0;
}
