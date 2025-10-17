/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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

// template<class In, class Out>
// concept indirectly_copyable_storable;

#include <uscl/std/iterator>

template <class I, class O>
  requires cuda::std::indirectly_copyable<I, O>
__host__ __device__ constexpr bool indirectly_copyable_storable_subsumption()
{
  return false;
}

template <class I, class O>
  requires cuda::std::indirectly_copyable_storable<I, O>
__host__ __device__ constexpr bool indirectly_copyable_storable_subsumption()
{
  return true;
}

#ifndef __NVCOMPILER // nvbug 3885350
static_assert(indirectly_copyable_storable_subsumption<int*, int*>(), "");
#endif

int main(int, char**)
{
  return 0;
}
