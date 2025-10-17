/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

// Mandates: invoke result must fail to compile when used with device lambdas.
// UNSUPPORTED: clang && (!nvcc)

// <cuda/std/functional>

// result_of<Fn(ArgTypes...)>

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class Ret, class Fn>
__host__ __device__ void test_lambda(Fn&&)
{
  static_assert(cuda::std::is_same_v<Ret, typename cuda::std::result_of<Fn()>::type>);

  static_assert(cuda::std::is_same_v<Ret, typename cuda::std::invoke_result<Fn>::type>);
}

int main(int, char**)
{
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
  { // extended device lambda
    test_lambda<int>([] __device__() {
      return 42;
    });
    test_lambda<double>([] __device__() {
      return 42.0;
    });
  }
#endif // TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)

  return 0;
}
