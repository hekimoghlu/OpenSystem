/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

#include <uscl/std/concepts>
#include <uscl/std/ranges>
#include <uscl/std/string_view>

static_assert(
  cuda::std::same_as<cuda::std::ranges::iterator_t<cuda::std::string_view>, cuda::std::string_view::iterator>);
static_assert(cuda::std::ranges::common_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::random_access_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::contiguous_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::view<cuda::std::string_view>
              && cuda::std::ranges::enable_view<cuda::std::string_view>);
static_assert(cuda::std::ranges::sized_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::string_view>);
static_assert(cuda::std::ranges::viewable_range<cuda::std::string_view>);

static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<cuda::std::string_view const>,
                                 cuda::std::string_view::const_iterator>);
static_assert(cuda::std::ranges::common_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::random_access_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::contiguous_range<cuda::std::string_view const>);
static_assert(!cuda::std::ranges::view<cuda::std::string_view const>
              && !cuda::std::ranges::enable_view<cuda::std::string_view const>);
static_assert(cuda::std::ranges::sized_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::string_view const>);
static_assert(cuda::std::ranges::viewable_range<cuda::std::string_view const>);

int main(int, char**)
{
  return 0;
}
