/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

// UNSUPPORTED: no-localization
// XFAIL: true

// <random>

// template<class RealType = double>
// class uniform_real_distribution

// template <class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os,
//            const uniform_real_distribution& x);
//
// template <class charT, class traits>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is,
//            uniform_real_distribution& x);

#include <uscl/std/__random_>
#include <uscl/std/cassert>
#include <uscl/std/sstream>

#include "test_macros.h"

__host__ __device__ void test()
{
  using D = cuda::std::uniform_real_distribution<>;
  D d1(3, 8);
  cuda::std::ostringstream os;
  os << d1;
  cuda::std::istringstream is(os.str());
  D d2;
  is >> d2;
  assert(d1 == d2);
}

int main(int, char**)
{
  test();
  return 0;
}
