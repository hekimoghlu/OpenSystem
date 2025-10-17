/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

// <chrono>

// using years = duration<signed integer type of at least 17 bits, ratio_multiply<ratio<146097, 400>, days::period>>

#include <uscl/std/chrono>
#include <uscl/std/limits>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::chrono::years D;
  typedef D::rep Rep;
  typedef D::period Period;
  static_assert(cuda::std::is_signed<Rep>::value, "");
  static_assert(cuda::std::is_integral<Rep>::value, "");
  static_assert(cuda::std::numeric_limits<Rep>::digits >= 17, "");
  static_assert(
    cuda::std::is_same_v<Period,
                         cuda::std::ratio_multiply<cuda::std::ratio<146097, 400>, cuda::std::chrono::days::period>>,
    "");

  return 0;
}
