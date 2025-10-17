/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/cassert>
#include <uscl/std/complex>

#include <nv/target>

#include <complex>

#include "test_macros.h"

template <class T, class U>
void test_comparison()
{
  ::cuda::std::complex<T> input{static_cast<T>(-1.0), static_cast<T>(1.0)};

  const ::std::complex<U> not_equal_real{static_cast<T>(-1.0), 0};
  const ::std::complex<U> not_equal_imag{0, static_cast<T>(1.0)};
  const ::std::complex<U> equal{static_cast<T>(-1.0), static_cast<T>(1.0)};

  assert(!(input == not_equal_real));
  assert(!(input == not_equal_imag));
  assert(input == equal);

  assert(input != not_equal_real);
  assert(input != not_equal_imag);
  assert(!(input != equal));
}

void test()
{
  test_comparison<float, float>();
  test_comparison<double, float>();
  test_comparison<double, double>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();));

  return 0;
}
