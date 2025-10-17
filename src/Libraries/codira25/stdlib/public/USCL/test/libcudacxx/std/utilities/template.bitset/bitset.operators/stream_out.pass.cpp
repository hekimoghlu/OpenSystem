/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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

// UNSUPPORTED: no-localization

// test:

// template <class charT, class traits, size_t N>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os, const bitset<N>& x);

#include <uscl/std/version>

#ifndef _LIBCUDACXX_HAS_SSTREAM
int main(int, char**)
{
  return 0;
}
#else

#  include <cuda/std/bitset>
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  cuda::std::ostringstream os;
  cuda::std::bitset<8> b(0x5A);
  os << b;
  assert(os.str() == "01011010");

  return 0;
}

#endif
