/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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

// <random>

// template<class RealType = double>
// class uniform_real_distribution

// explicit uniform_real_distribution(RealType a = 0.0,
//                                    RealType b = 1.0);             // before C++20
// uniform_real_distribution() : uniform_real_distribution(0.0) {}   // C++20
// explicit uniform_real_distribution(RealType a, RealType b = 1.0); // C++20

#include <uscl/std/__random_>
#include <uscl/std/cassert>

#include "make_implicit.h"
#include "test_convertible.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_implicit()
{
  using D = cuda::std::uniform_real_distribution<T>;
  static_assert(test_convertible<D>(), "");
  assert(D(0) == make_implicit<D>());
  static_assert(!test_convertible<D, T>(), "");
  static_assert(!test_convertible<D, T, T>(), "");
}

__host__ __device__ void test()
{
  {
    using D = cuda::std::uniform_real_distribution<>;
    D d;
    assert(d.a() == 0.0);
    assert(d.b() == 1.0);
  }
  {
    using D = cuda::std::uniform_real_distribution<>;
    D d(-6.5);
    assert(d.a() == -6.5);
    assert(d.b() == 1.0);
  }
  {
    using D = cuda::std::uniform_real_distribution<>;
    D d(-6.9, 106.1);
    assert(d.a() == -6.9);
    assert(d.b() == 106.1);
  }
}

int main(int, char**)
{
  test();
  test_implicit<float>();
  test_implicit<double>();

  return 0;
}
