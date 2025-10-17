/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<range R>
// using range_difference_t = iter_difference_t<iterator_t<R>>;

// template<range R>
// using range_value_t = iter_value_t<iterator_t<R>>;

// template<range R>
// using range_reference_t = iter_reference_t<iterator_t<R>>;

// template<range R>
// using range_rvalue_reference_t = iter_rvalue_reference_t<iterator_t<R>>;

#include <uscl/std/concepts>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

static_assert(cuda::std::same_as<cuda::std::ranges::range_difference_t<test_range<cpp20_input_iterator>>,
                                 cuda::std::iter_difference_t<int*>>,
              "");
static_assert(
  cuda::std::same_as<cuda::std::ranges::range_value_t<test_range<cpp20_input_iterator>>, cuda::std::iter_value_t<int*>>,
  "");
static_assert(cuda::std::same_as<cuda::std::ranges::range_reference_t<test_range<cpp20_input_iterator>>,
                                 cuda::std::iter_reference_t<int*>>,
              "");
static_assert(cuda::std::same_as<cuda::std::ranges::range_rvalue_reference_t<test_range<cpp20_input_iterator>>,
                                 cuda::std::iter_rvalue_reference_t<int*>>,
              "");
static_assert(cuda::std::same_as<cuda::std::ranges::range_common_reference_t<test_range<cpp20_input_iterator>>,
                                 cuda::std::iter_common_reference_t<int*>>,
              "");

int main(int, char**)
{
  return 0;
}
