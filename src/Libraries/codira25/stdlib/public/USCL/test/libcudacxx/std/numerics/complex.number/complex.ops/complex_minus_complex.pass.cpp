/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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

// template<class T>
//   complex<T>
//   operator-(const complex<T>& lhs, const complex<T>& rhs);

#include <uscl/std/cassert>
#include <uscl/std/complex>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool test()
{
  {
    cuda::std::complex<T> lhs(1.5, 2.5);
    cuda::std::complex<T> rhs(3.5, 4.5);
    cuda::std::complex<T> x(-2.0, -2.0);
    assert(lhs - rhs == x);
  }
  {
    cuda::std::complex<T> lhs(1.5, -2.5);
    cuda::std::complex<T> rhs(-3.5, 4.5);
    cuda::std::complex<T> x(5.0, -7.0);
    assert(lhs - rhs == x);
  }

  return true;
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
  static_assert(test<float>(), "");
  static_assert(test<double>(), "");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(test<long double>(), "");
#endif // _CCCL_HAS_LONG_DOUBLE()

  return 0;
}
