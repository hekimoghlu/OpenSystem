/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

// <span>

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <uscl/std/cstddef>
#include <uscl/std/span>

template <class T, size_t extent>
__host__ __device__ cuda::std::span<T, extent> createImplicitSpan(T* ptr, size_t len)
{
  return {ptr, len}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(arr, 3);

  cuda::std::span<int> sp = {0, 0}; // expected-error {{no matching constructor for initialization of
                                    // 'cuda::std::span<int>'}}
  cuda::std::span<int, 2> sp2 = {0, 0}; // expected-error {{no matching constructor for initialization of
                                        // 'cuda::std::span<int, 2>'}}
  cuda::std::span<const int> csp = {0, 0}; // expected-error {{no matching constructor for initialization of
                                           // 'cuda::std::span<const int>'}}
  cuda::std::span<const int, 2> csp2 = {0, 0}; // expected-error {{no matching constructor for initialization of
                                               // 'cuda::std::span<const int, 2>'}}

  return 0;
}
