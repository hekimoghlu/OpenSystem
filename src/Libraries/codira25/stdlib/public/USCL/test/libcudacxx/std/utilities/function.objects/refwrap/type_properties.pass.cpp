/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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

// <functional>

// reference_wrapper

// Test that reference wrapper meets the requirements of CopyConstructible and
// CopyAssignable, and TriviallyCopyable (starting in C++14).

// #include <uscl/std/functional>
#include <uscl/std/type_traits>
#include <uscl/std/utility>
#ifdef _LIBCUDACXX_HAS_
#  include <cuda/std/string>
#endif

#include "test_macros.h"

class MoveOnly
{
  __host__ __device__ MoveOnly(const MoveOnly&);
  __host__ __device__ MoveOnly& operator=(const MoveOnly&);

  int data_;

public:
  __host__ __device__ MoveOnly(int data = 1)
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
};

template <class T>
__host__ __device__ void test()
{
  typedef cuda::std::reference_wrapper<T> Wrap;
  static_assert(cuda::std::is_copy_constructible<Wrap>::value, "");
  static_assert(cuda::std::is_copy_assignable<Wrap>::value, "");
  static_assert(cuda::std::is_trivially_copyable<Wrap>::value, "");
}

int main(int, char**)
{
  test<int>();
  test<double>();
#ifdef _LIBCUDACXX_HAS_
  test<cuda::std::string>();
#endif
  test<MoveOnly>();

  return 0;
}
