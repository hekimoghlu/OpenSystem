/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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

#include <uscl/std/type_traits>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

//  Assumption: minValue < maxValue
//  Assumption: minValue <= rhs <= maxValue
//  Assumption: minValue <= lhs <= maxValue
//  Assumption: minValue >= 0
template <typename T, T minValue, T maxValue>
__host__ __device__ T euclidian_addition(T rhs, T lhs)
{
  const T modulus = maxValue - minValue + 1;
  T ret           = rhs + lhs;
  if (ret > maxValue)
  {
    ret -= modulus;
  }
  return ret;
}

//  Assumption: minValue < maxValue
//  Assumption: minValue <= rhs <= maxValue
//  Assumption: minValue <= lhs <= maxValue
//  Assumption: minValue >= 0
template <typename T, T minValue, T maxValue>
__host__ __device__ T euclidian_subtraction(T lhs, T rhs)
{
  const T modulus = maxValue - minValue + 1;
  T ret           = lhs - rhs;
  if constexpr (cuda::std::is_signed_v<T>) // avoids warning about comparison with zero if T is unsigned
  {
    if (ret < minValue)
    {
      ret += modulus;
    }
  }
  if (ret > maxValue) // this can happen if T is unsigned
  {
    ret += modulus;
  }
  return ret;
}
