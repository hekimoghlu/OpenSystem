/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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

// UNSUPPORTED: msvc

// <utility>

// template <class T1, class T2> struct pair

// struct piecewise_construct_t { explicit piecewise_construct_t() = default; };
// constexpr piecewise_construct_t piecewise_construct = piecewise_construct_t();

#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/utility>

#include "test_macros.h"

class A
{
  int i_;
  char c_;

public:
  __host__ __device__ A(int i, char c)
      : i_(i)
      , c_(c)
  {}
  __host__ __device__ int get_i() const
  {
    return i_;
  }
  __host__ __device__ char get_c() const
  {
    return c_;
  }
};

class B
{
  double d_;
  unsigned u1_;
  unsigned u2_;

public:
  __host__ __device__ B(double d, unsigned u1, unsigned u2)
      : d_(d)
      , u1_(u1)
      , u2_(u2)
  {}
  __host__ __device__ double get_d() const
  {
    return d_;
  }
  __host__ __device__ unsigned get_u1() const
  {
    return u1_;
  }
  __host__ __device__ unsigned get_u2() const
  {
    return u2_;
  }
};

int main(int, char**)
{
  cuda::std::pair<A, B> p(
    cuda::std::piecewise_construct, cuda::std::make_tuple(4, 'a'), cuda::std::make_tuple(3.5, 6u, 2u));
  assert(p.first.get_i() == 4);
  assert(p.first.get_c() == 'a');
  assert(p.second.get_d() == 3.5);
  assert(p.second.get_u1() == 6u);
  assert(p.second.get_u2() == 2u);

  return 0;
}
