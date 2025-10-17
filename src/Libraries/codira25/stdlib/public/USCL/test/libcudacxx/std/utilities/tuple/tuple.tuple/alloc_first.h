/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

#ifndef ALLOC_FIRST_H
#define ALLOC_FIRST_H

#include <uscl/std/cassert>

#include "allocators.h"

struct alloc_first
{
  STATIC_MEMBER_VAR(allocator_constructed, bool)

  using allocator_type = A1<int>;

  int data_;

  __host__ __device__ alloc_first()
      : data_(0)
  {}
  __host__ __device__ alloc_first(int d)
      : data_(d)
  {}
  __host__ __device__ alloc_first(cuda::std::allocator_arg_t, const A1<int>& a)
      : data_(0)
  {
    assert(a.id() == 5);
    allocator_constructed() = true;
  }

  __host__ __device__ alloc_first(cuda::std::allocator_arg_t, const A1<int>& a, int d)
      : data_(d)
  {
    assert(a.id() == 5);
    allocator_constructed() = true;
  }

  __host__ __device__ alloc_first(cuda::std::allocator_arg_t, const A1<int>& a, const alloc_first& d)
      : data_(d.data_)
  {
    assert(a.id() == 5);
    allocator_constructed() = true;
  }

  __host__ __device__ ~alloc_first()
  {
    data_ = -1;
  }

  __host__ __device__ friend bool operator==(const alloc_first& x, const alloc_first& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend bool operator<(const alloc_first& x, const alloc_first& y)
  {
    return x.data_ < y.data_;
  }
};

#endif // ALLOC_FIRST_H
