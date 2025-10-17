/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#include <uscl/std/cstddef>

#include "test_macros.h"
// #include <functional>

class MoveOnly
{
  int data_;

public:
  __host__ __device__ constexpr MoveOnly(int data = 1)
      : data_(data)
  {}

  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  __host__ __device__ constexpr MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  __host__ __device__ constexpr MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  __host__ __device__ constexpr int get() const
  {
    return data_;
  }

  __host__ __device__ friend constexpr bool operator==(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend constexpr bool operator!=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ != y.data_;
  }
  __host__ __device__ friend constexpr bool operator<(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ < y.data_;
  }
  __host__ __device__ friend constexpr bool operator<=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ <= y.data_;
  }
  __host__ __device__ friend constexpr bool operator>(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ > y.data_;
  }
  __host__ __device__ friend constexpr bool operator>=(const MoveOnly& x, const MoveOnly& y)
  {
    return x.data_ >= y.data_;
  }

#if TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  __host__ __device__ friend constexpr auto operator<=>(const MoveOnly&, const MoveOnly&) = default;
#endif // TEST_STD_VER > 2017 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  __host__ __device__ constexpr MoveOnly operator+(const MoveOnly& x) const
  {
    return MoveOnly(data_ + x.data_);
  }
  __host__ __device__ constexpr MoveOnly operator*(const MoveOnly& x) const
  {
    return MoveOnly(data_ * x.data_);
  }

  template <class T>
  void operator,(T const&) = delete;
};

/*
template <>
struct cuda::std::hash<MoveOnly>
{
    typedef MoveOnly argument_type;
    typedef size_t result_type;
    __host__ __device__ constexpr size_t operator()(const MoveOnly& x) const {return x.get();}
};
*/

#endif // MOVEONLY_H
