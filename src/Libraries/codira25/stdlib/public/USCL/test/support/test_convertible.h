/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_CONVERTIBLE_H
#define SUPPORT_TEST_CONVERTIBLE_H

// "test_convertible<Tp, Args...>()" is a metafunction used to check if 'Tp'
// is implicitly convertible from 'Args...' for any number of arguments,
// Unlike 'std::is_convertible' which only allows checking for single argument
// conversions.

#include <uscl/std/type_traits>

#include "test_macros.h"

namespace detail
{
template <class Tp>
__host__ __device__ void eat_type(Tp);

template <class Tp, class... Args>
__host__ __device__ constexpr auto test_convertible_imp(int)
  -> decltype(eat_type<Tp>({cuda::std::declval<Args>()...}), true)
{
  return true;
}

template <class Tp, class... Args>
__host__ __device__ constexpr auto test_convertible_imp(long) -> bool
{
  return false;
}
} // namespace detail

template <class Tp, class... Args>
__host__ __device__ constexpr bool test_convertible()
{
  return detail::test_convertible_imp<Tp, Args...>(0);
}

#endif // SUPPORT_TEST_CONVERTIBLE_H
