/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
// UNSUPPORTED: c++17

// <span>

// template <class It, class End>
// constexpr explicit(Extent != dynamic_extent) span(It first, End last);
// Requires: [first, last) shall be a valid range.
//   If Extent is not equal to dynamic_extent, then last - first shall be equal to Extent.
//

#include <uscl/std/iterator>
#include <uscl/std/span>

#include "test_macros.h"

template <class T, size_t Extent>
cuda::std::span<T, Extent> createImplicitSpan(T* first, T* last)
{
  return {first, last}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(cuda::std::begin(arr), cuda::std::end(arr));

  return 0;
}
