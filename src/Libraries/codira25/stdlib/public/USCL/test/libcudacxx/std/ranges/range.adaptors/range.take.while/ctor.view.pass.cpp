/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr take_while_view(V base, Pred pred);

#include <uscl/std/cassert>
#include <uscl/std/ranges>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  MoveOnly mo;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  bool copied = false;
  bool moved  = false;
  Pred()      = default;
  __host__ __device__ constexpr Pred(Pred&&)
      : moved(true)
  {}
  __host__ __device__ constexpr Pred(const Pred&)
      : copied(true)
  {}
  __host__ __device__ bool operator()(int) const;
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::take_while_view<View, Pred> twv = {View{{}, MoveOnly{5}}, Pred{}};
    assert(twv.pred().moved);
    assert(!twv.pred().copied);
    assert(cuda::std::move(twv).base().mo.get() == 5);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
