/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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

// test that <cuda/std/bitset> includes <cuda/std/string> and <cuda/std/iosfwd>

#include <uscl/std/bitset>

#include "test_macros.h"

template <class>
__host__ __device__ void test_typedef()
{}

int main(int, char**)
{
#ifdef _LIBCUDACXX_HAS_STRING
  { // test for <cuda/std/string>
    [[maybe_unused]] cuda::std::string s;
  }
#endif
  { // test for <cuda/std/iosfwd>
    test_typedef<cuda::std::ios>();
    test_typedef<cuda::std::istream>();
    test_typedef<cuda::std::ostream>();
    test_typedef<cuda::std::iostream>();
  }

  return 0;
}
