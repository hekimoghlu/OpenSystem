/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

// test ratio typedef's

#include <uscl/std/ratio>

int main(int, char**)
{
  static_assert(cuda::std::atto::num == 1 && cuda::std::atto::den == 1000000000000000000ULL, "");
  static_assert(cuda::std::femto::num == 1 && cuda::std::femto::den == 1000000000000000ULL, "");
  static_assert(cuda::std::pico::num == 1 && cuda::std::pico::den == 1000000000000ULL, "");
  static_assert(cuda::std::nano::num == 1 && cuda::std::nano::den == 1000000000ULL, "");
  static_assert(cuda::std::micro::num == 1 && cuda::std::micro::den == 1000000ULL, "");
  static_assert(cuda::std::milli::num == 1 && cuda::std::milli::den == 1000ULL, "");
  static_assert(cuda::std::centi::num == 1 && cuda::std::centi::den == 100ULL, "");
  static_assert(cuda::std::deci::num == 1 && cuda::std::deci::den == 10ULL, "");
  static_assert(cuda::std::deca::num == 10ULL && cuda::std::deca::den == 1, "");
  static_assert(cuda::std::hecto::num == 100ULL && cuda::std::hecto::den == 1, "");
  static_assert(cuda::std::kilo::num == 1000ULL && cuda::std::kilo::den == 1, "");
  static_assert(cuda::std::mega::num == 1000000ULL && cuda::std::mega::den == 1, "");
  static_assert(cuda::std::giga::num == 1000000000ULL && cuda::std::giga::den == 1, "");
  static_assert(cuda::std::tera::num == 1000000000000ULL && cuda::std::tera::den == 1, "");
  static_assert(cuda::std::peta::num == 1000000000000000ULL && cuda::std::peta::den == 1, "");
  static_assert(cuda::std::exa::num == 1000000000000000000ULL && cuda::std::exa::den == 1, "");

  return 0;
}
