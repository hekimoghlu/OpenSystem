/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
//
// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept

// An overload exists for each of char and all arithmetic types except bool.

#include <uscl/std/numeric>

#include "test_macros.h"

__host__ __device__ int func1()
{
  return 1;
}
__host__ __device__ int func2()
{
  return 2;
}

struct Incomplete;
Incomplete* ip = nullptr;
void* vp       = nullptr;

int main(int, char**)
{
  (void) cuda::std::midpoint(false, true); // expected-error {{no matching function for call to 'midpoint'}}

  //  A couple of odd pointer types that should fail
  (void) cuda::std::midpoint(nullptr, nullptr); // expected-error {{no matching function for call to 'midpoint'}}
  (void) cuda::std::midpoint(func1, func2); // expected-error {{no matching function for call to 'midpoint'}}
  (void) cuda::std::midpoint(ip, ip); // expected-error {{no matching function for call to 'midpoint'}}
  (void) cuda::std::midpoint(vp, vp); // expected-error {{no matching function for call to 'midpoint'}}

  return 0;
}
