/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

// UNSUPPORTED: c++17

// template<class S, class I>
// concept sentinel_for;

#include <uscl/std/array>
#include <uscl/std/concepts>
#include <uscl/std/iterator>

// clang-format off
template<cuda::std::input_or_output_iterator, cuda::std::semiregular>
__host__ __device__ constexpr bool check_sentinel_subsumption() {
  return false;
}

template<class I, cuda::std::sentinel_for<I> >
__host__ __device__ constexpr bool check_subsumption() {
  return true;
}
// clang-format on

static_assert(check_subsumption<cuda::std::array<int, 4>::iterator, cuda::std::array<int, 4>::iterator>(), "");

int main(int, char**)
{
  return 0;
}
