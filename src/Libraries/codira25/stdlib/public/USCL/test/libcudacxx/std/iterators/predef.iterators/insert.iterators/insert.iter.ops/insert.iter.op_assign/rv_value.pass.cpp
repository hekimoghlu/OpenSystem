/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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

// <cuda/std/iterator>

// insert_iterator

// requires CopyConstructible<Cont::value_type>
//   insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <uscl/std/cassert>
#include <uscl/std/inplace_vector>
#include <uscl/std/iterator>
#include <uscl/std/memory>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

template <class C>
__host__ __device__ void
test(C c1,
     typename C::difference_type j,
     typename C::value_type x1,
     typename C::value_type x2,
     typename C::value_type x3,
     const C& c2)
{
  cuda::std::insert_iterator<C> q(c1, c1.begin() + j);
  q = cuda::std::move(x1);
  q = cuda::std::move(x2);
  q = cuda::std::move(x3);
  assert(c1 == c2);
}

template <class C>
__host__ __device__ void
insert3at(C& c, typename C::iterator i, typename C::value_type x1, typename C::value_type x2, typename C::value_type x3)
{
  i = c.insert(i, cuda::std::move(x1));
  i = c.insert(++i, cuda::std::move(x2));
  c.insert(++i, cuda::std::move(x3));
}

int main(int, char**)
{
  {
    typedef cuda::std::inplace_vector<MoveOnly, 10> C;
    C c1;
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(i);
    }
    C c2;
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(i);
    }
    insert3at(c2, c2.begin(), 3, 4, 5);
    test(cuda::std::move(c1), 0, 3, 4, 5, c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(i);
    }
    c2.clear();
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(i);
    }
    insert3at(c2, c2.begin() + 1, 3, 4, 5);
    test(cuda::std::move(c1), 1, 3, 4, 5, c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(i);
    }
    c2.clear();
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(i);
    }
    insert3at(c2, c2.begin() + 2, 3, 4, 5);
    test(cuda::std::move(c1), 2, 3, 4, 5, c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(i);
    }
    c2.clear();
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(i);
    }
    insert3at(c2, c2.begin() + 3, 3, 4, 5);
    test(cuda::std::move(c1), 3, 3, 4, 5, c2);
  }

  return 0;
}
