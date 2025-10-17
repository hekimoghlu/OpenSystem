/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

// test ratio:  The static data members num and den shall have the common
//    divisor of the absolute values of N and D:

#include <uscl/std/ratio>

template <long long N, long long D, long long eN, long long eD>
__host__ __device__ void test()
{
  static_assert((cuda::std::ratio<N, D>::num == eN), "");
  static_assert((cuda::std::ratio<N, D>::den == eD), "");
}

int main(int, char**)
{
  test<1, 1, 1, 1>();
  test<1, 10, 1, 10>();
  test<10, 10, 1, 1>();
  test<10, 1, 10, 1>();
  test<12, 4, 3, 1>();
  test<12, -4, -3, 1>();
  test<-12, 4, -3, 1>();
  test<-12, -4, 3, 1>();
  test<4, 12, 1, 3>();
  test<4, -12, -1, 3>();
  test<-4, 12, -1, 3>();
  test<-4, -12, 1, 3>();
  test<222, 333, 2, 3>();
  test<222, -333, -2, 3>();
  test<-222, 333, -2, 3>();
  test<-222, -333, 2, 3>();
  test<0x7FFFFFFFFFFFFFFFLL, 127, 72624976668147841LL, 1>();
  test<-0x7FFFFFFFFFFFFFFFLL, 127, -72624976668147841LL, 1>();
  test<0x7FFFFFFFFFFFFFFFLL, -127, -72624976668147841LL, 1>();
  test<-0x7FFFFFFFFFFFFFFFLL, -127, 72624976668147841LL, 1>();

  return 0;
}
