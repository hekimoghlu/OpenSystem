/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

// <cuda/std/format>

// cuda::std::__fmt_arg_t

#include <uscl/std/__format_>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<cuda::std::__fmt_arg_t>, cuda::std::uint8_t>);

static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__none) == 0);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__boolean) == 1);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__char_type) == 2);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__int) == 3);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__long_long) == 4);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__unsigned) == 5);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__unsigned_long_long) == 6);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__float) == 7);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__double) == 8);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__long_double) == 9);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__const_char_type_ptr) == 10);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__string_view) == 11);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__ptr) == 12);
static_assert(cuda::std::uint8_t(cuda::std::__fmt_arg_t::__handle) == 13);

int main(int, char**)
{
  return 0;
}
