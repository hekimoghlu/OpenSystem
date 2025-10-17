/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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

// <cuda/std/complex>

// template<Arithmetic T>
//   T
//   norm(T x);

#include <uscl/std/cassert>
#include <uscl/std/complex>
#include <uscl/std/type_traits>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T x, typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::norm(x)), double>::value), "");
  assert(cuda::std::norm(x) == norm(cuda::std::complex<double>(static_cast<double>(x), 0)));
}

template <class T>
__host__ __device__ void test(T x, typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::norm(x)), T>::value), "");
  assert(cuda::std::norm(x) == norm(cuda::std::complex<T>(x, 0)));
}

template <class T>
__host__ __device__ void test()
{
  test<T>(0);
  test<T>(1);
  test<T>(10);
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  test<int>();
  test<unsigned>();
  test<long long>();

  return 0;
}
