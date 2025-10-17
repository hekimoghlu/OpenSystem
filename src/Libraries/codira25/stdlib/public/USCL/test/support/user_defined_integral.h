/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#ifndef SUPPORT_USER_DEFINED_INTEGRAL_H
#define SUPPORT_USER_DEFINED_INTEGRAL_H

template <class T>
struct UserDefinedIntegral
{
  __host__ __device__ constexpr UserDefinedIntegral()
      : value(0)
  {}
  __host__ __device__ constexpr UserDefinedIntegral(T v)
      : value(v)
  {}
  __host__ __device__ constexpr operator T() const
  {
    return value;
  }
  T value;
};

// Poison the arithmetic and comparison operations
template <class T, class U>
__host__ __device__ constexpr void operator+(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator-(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator*(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator/(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator==(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator!=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator<(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator>(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator<=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

template <class T, class U>
__host__ __device__ constexpr void operator>=(UserDefinedIntegral<T>, UserDefinedIntegral<U>);

#endif // SUPPORT_USER_DEFINED_INTEGRAL_H
