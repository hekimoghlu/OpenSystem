/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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

// template<class In>
// concept indirectly_readable;

#include "../incrementable.h"

#include <uscl/std/concepts>
#include <uscl/std/iterator>

static_assert(cuda::std::incrementable<int>, "");
static_assert(cuda::std::incrementable<int*>, "");
static_assert(cuda::std::incrementable<int**>, "");

static_assert(!cuda::std::incrementable<postfix_increment_returns_void>, "");
static_assert(!cuda::std::incrementable<postfix_increment_returns_copy>, "");
static_assert(!cuda::std::incrementable<has_integral_minus>, "");
static_assert(!cuda::std::incrementable<has_distinct_difference_type_and_minus>, "");
static_assert(!cuda::std::incrementable<missing_difference_type>, "");
static_assert(!cuda::std::incrementable<floating_difference_type>, "");
static_assert(!cuda::std::incrementable<non_const_minus>, "");
static_assert(!cuda::std::incrementable<non_integral_minus>, "");
static_assert(!cuda::std::incrementable<bad_difference_type_good_minus>, "");
static_assert(!cuda::std::incrementable<not_default_initializable>, "");
static_assert(!cuda::std::incrementable<not_movable>, "");
static_assert(!cuda::std::incrementable<preinc_not_declared>, "");
static_assert(!cuda::std::incrementable<postinc_not_declared>, "");
static_assert(cuda::std::incrementable<incrementable_with_difference_type>, "");
static_assert(cuda::std::incrementable<incrementable_without_difference_type>, "");
static_assert(cuda::std::incrementable<difference_type_and_void_minus>, "");
static_assert(!cuda::std::incrementable<noncopyable_with_difference_type>, "");
static_assert(!cuda::std::incrementable<noncopyable_without_difference_type>, "");
static_assert(!cuda::std::incrementable<noncopyable_with_difference_type_and_minus>, "");

int main(int, char**)
{
  return 0;
}
