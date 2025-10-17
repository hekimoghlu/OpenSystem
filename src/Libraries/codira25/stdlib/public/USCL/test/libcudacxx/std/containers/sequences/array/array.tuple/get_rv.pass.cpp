/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> T&& get(array<T, N>&& a);

#include <uscl/std/array>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

struct MoveOnly
{
  double val_ = 0.0;

  MoveOnly()                      = default;
  MoveOnly(MoveOnly&&)            = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  __host__ __device__ constexpr MoveOnly(const double val) noexcept
      : val_(val)
  {}
};

int main(int, char**)
{
  {
    typedef cuda::std::array<MoveOnly, 1> C;
    C c = {3.5};
    static_assert(cuda::std::is_same<MoveOnly&&, decltype(cuda::std::get<0>(cuda::std::move(c)))>::value, "");
    MoveOnly t = cuda::std::get<0>(cuda::std::move(c));
    assert(t.val_ == 3.5);
  }

  return 0;
}
