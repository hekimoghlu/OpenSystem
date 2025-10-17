/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
//   polar(const T& rho, const T& theta = T());  // changed from '0' by LWG#2870

#include <uscl/std/cassert>
#include <uscl/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const T& rho, cuda::std::complex<T> x)
{
  assert(cuda::std::polar(rho) == x);
}

template <class T>
__host__ __device__ void test(const T& rho, const T& theta, cuda::std::complex<T> x)
{
  assert(cuda::std::polar(rho, theta) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(T(0), cuda::std::complex<T>(0, 0));
  test(T(1), cuda::std::complex<T>(1, 0));
  test(T(100), cuda::std::complex<T>(100, 0));
  test(T(0), T(0), cuda::std::complex<T>(0, 0));
  test(T(1), T(0), cuda::std::complex<T>(1, 0));
  test(T(100), T(0), cuda::std::complex<T>(100, 0));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    T r                     = real(testcases[i]);
    T theta                 = imag(testcases[i]);
    cuda::std::complex<T> z = cuda::std::polar(r, theta);
    switch (classify(r))
    {
      case zero:
        if (cuda::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
        {
          int c = classify(z);
          assert(c == NaN || c == non_zero_nan);
        }
        else
        {
          assert(z == cuda::std::complex<T>());
        }
        break;
      case non_zero:
        if (cuda::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
        {
          int c = classify(z);
          assert(c == NaN || c == non_zero_nan);
        }
        else
        {
          printf("in: %f %f\n", float(testcases[i].real()), float(testcases[i].imag()));
          is_about(cuda::std::abs(z), r);
        }
        break;
      case inf:
        if (r < T(0))
        {
          int c = classify(z);
          assert(c == NaN || c == non_zero_nan);
        }
        else
        {
          assert(classify(z) == inf);
          if (classify(theta) != NaN && classify(theta) != inf)
          {
            assert(classify(real(z)) != NaN);
            assert(classify(imag(z)) != NaN);
          }
        }
        break;
      case NaN:
      case non_zero_nan: {
        int c = classify(z);
        assert(c == NaN || c == non_zero_nan);
      }
      break;
    }
  }
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
  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_edges<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}
