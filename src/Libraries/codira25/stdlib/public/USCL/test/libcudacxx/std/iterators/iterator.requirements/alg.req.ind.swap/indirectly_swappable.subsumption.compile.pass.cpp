/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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

// UNSUPPORTED: c++17

// template<class I1, class I2>
// concept indirectly_swappable;

#include <uscl/std/concepts>
#include <uscl/std/iterator>

template <class I1, class I2>
  requires cuda::std::indirectly_readable<I1> && cuda::std::indirectly_readable<I2>
__host__ __device__ constexpr bool indirectly_swappable_subsumption()
{
  return false;
}

template <class I1, class I2>
  requires cuda::std::indirectly_swappable<I1, I2>
__host__ __device__ constexpr bool indirectly_swappable_subsumption()
{
  return true;
}

static_assert(indirectly_swappable_subsumption<int*, int*>(), "");

int main(int, char**)
{
  return 0;
}
