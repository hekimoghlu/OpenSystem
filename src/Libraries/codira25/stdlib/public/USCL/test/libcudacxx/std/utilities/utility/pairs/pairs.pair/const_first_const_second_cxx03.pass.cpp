/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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

// <utility>

// template <class T1, class T2> struct pair

// pair(const T1& x, const T2& y);

#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

class A
{
  int data_;

public:
  __host__ __device__ A(int data)
      : data_(data)
  {}

  __host__ __device__ bool operator==(const A& a) const
  {
    return data_ == a.data_;
  }
};

int main(int, char**)
{
  {
    typedef cuda::std::pair<float, short*> P;
    P p(3.5f, 0);
    assert(p.first == 3.5f);
    assert(p.second == nullptr);
  }
  {
    typedef cuda::std::pair<A, int> P;
    P p(1, 2);
    assert(p.first == A(1));
    assert(p.second == 2);
  }

  return 0;
}
