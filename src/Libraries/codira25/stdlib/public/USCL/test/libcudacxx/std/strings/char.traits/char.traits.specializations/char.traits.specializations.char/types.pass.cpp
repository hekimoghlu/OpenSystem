/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

#include <uscl/std/__string_>
#include <uscl/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::char_type, char>);
static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::int_type, int>);
// static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::off_type, cuda::std::streamoff>);
// static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::pos_type, cuda::std::streampos>);
// static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::state_type, cuda::std::mbstate_t>);
// static_assert(std::is_same_v<std::char_traits<char>::comparison_category, std::strong_ordering>);

int main(int, char**)
{
  return 0;
}
