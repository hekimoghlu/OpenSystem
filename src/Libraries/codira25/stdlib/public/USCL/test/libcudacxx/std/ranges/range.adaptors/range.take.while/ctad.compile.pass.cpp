/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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

//   template<class R, class Pred>
//     take_while_view(R&&, Pred) -> take_while_view<views::all_t<R>, Pred>;

#include <uscl/std/cassert>
#include <uscl/std/ranges>
#include <uscl/std/utility>

#include "types.h"

struct Container
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Pred
{
  __host__ __device__ bool operator()(int i) const;
};

__host__ __device__ bool pred(int);

using owning_result = cuda::std::ranges::take_while_view<cuda::std::ranges::owning_view<Container>, Pred>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(Container{}, Pred{})), owning_result>);

using function_result = cuda::std::ranges::take_while_view<View, bool (*)(int)>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(View{}, pred)), function_result>);

using view_result = cuda::std::ranges::take_while_view<View, Pred>;
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(View{}, Pred{})), view_result>);

__host__ __device__ void testRef()
{
  Container c{};
  Pred p{};
  unused(c);
  unused(p);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::take_while_view(c, p)),
                                     cuda::std::ranges::take_while_view<cuda::std::ranges::ref_view<Container>, Pred>>);
}

int main(int, char**)
{
  return 0;
}
