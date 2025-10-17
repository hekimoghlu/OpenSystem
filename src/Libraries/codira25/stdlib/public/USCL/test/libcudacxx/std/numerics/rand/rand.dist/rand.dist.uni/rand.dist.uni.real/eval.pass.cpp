/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: long_tests

// <random>

// template<class RealType = double>
// class uniform_real_distribution

// template<class _URng> result_type operator()(_URng& g);

#include <uscl/std/__memory_>
#include <uscl/std/__random_>
#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/cmath>
#include <uscl/std/cstddef>
#include <uscl/std/numeric>
#include <uscl/std/span>

#include "test_macros.h"

template <class T>
__host__ __device__ inline T sqr(T x)
{
  return x * x;
}

constexpr int N = 10000;

template <class ResultType, class EngineType>
__host__ __device__ void test_statistics(cuda::std::span<ResultType, N> arr, ResultType a = 0.0, ResultType b = 0.0)
{
  using D = cuda::std::uniform_real_distribution<>;
  EngineType g;
  D d;
  for (int i = 0; i < N; ++i)
  {
    D::result_type v = d(g);
    assert(d.a() <= v && v < d.b());
    arr[i] = v;
  }
  D::result_type mean     = cuda::std::accumulate(arr.begin(), arr.end(), D::result_type(0)) / arr.size();
  D::result_type var      = 0;
  D::result_type skew     = 0;
  D::result_type kurtosis = 0;
  for (cuda::std::size_t i = 0; i < arr.size(); ++i)
  {
    D::result_type dbl = (arr[i] - mean);
    D::result_type d2  = sqr(dbl);
    var += d2;
    skew += dbl * d2;
    kurtosis += d2 * d2;
  }
  var /= arr.size();
  D::result_type dev = cuda::std::sqrt(var);
  skew /= arr.size() * dev * var;
  kurtosis /= arr.size() * var * var;
  kurtosis -= 3;
  D::result_type x_mean     = (d.a() + d.b()) / 2;
  D::result_type x_var      = sqr(d.b() - d.a()) / 12;
  D::result_type x_skew     = 0;
  D::result_type x_kurtosis = -6. / 5;
  assert(cuda::std::abs((mean - x_mean) / x_mean) < 0.01);
  assert(cuda::std::abs((var - x_var) / x_var) < 0.01);
  assert(cuda::std::abs(skew - x_skew) < 0.01);
  assert(cuda::std::abs((kurtosis - x_kurtosis) / x_kurtosis) < 0.01);
}

__host__ __device__ void test()
{
  cuda::std::unique_ptr<double[]> array = cuda::std::make_unique<double[]>(N);
  cuda::std::span<double, N> span{array.get(), array.get() + N};

  test_statistics<double, cuda::std::minstd_rand0>(span);
  test_statistics<double, cuda::std::minstd_rand0>(span, -1.0, 1.0);
#if 0 // not implemented
  test_statistics<double, cuda::std::minstd_rand>(span);
  test_statistics<double, cuda::std::minstd_rand>(span, 5.5, 25.0);
  test_statistics<double, cuda::std::mt19937>(span);
  test_statistics<double, cuda::std::mt19937_64>(span);
  test_statistics<double, cuda::std::ranlux24_base>(span);
  test_statistics<double, cuda::std::ranlux48_base>(span);
  test_statistics<double, cuda::std::ranlux24>(span);
  test_statistics<double, cuda::std::ranlux48>(span);
  test_statistics<double, cuda::std::knuth_b>(span);
#endif // not implemented
}

int main(int, char**)
{
  test();
  return 0;
}
