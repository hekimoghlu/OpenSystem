/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

/*
    This is testing an extension whereby only Types having an explicit conversion
    from UTypes are bound by the explicit tuple constructor.
*/

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "test_macros.h"

class MoveOnly
{
  MoveOnly(const MoveOnly&);
  MoveOnly& operator=(const MoveOnly&);

  int data_;

public:
  __host__ __device__ explicit MoveOnly(int data = 1)
      : data_(data)
  {}
  __host__ __device__ MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  __host__ __device__ MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  __host__ __device__ int get() const
  {
    return data_;
  }

  __host__ __device__ bool operator==(const MoveOnly& x) const
  {
    return data_ == x.data_;
  }
  __host__ __device__ bool operator<(const MoveOnly& x) const
  {
    return data_ < x.data_;
  }
};

int main(int, char**)
{
  {
    cuda::std::tuple<MoveOnly> t = 1;
    unused(t);
  }

  return 0;
}
