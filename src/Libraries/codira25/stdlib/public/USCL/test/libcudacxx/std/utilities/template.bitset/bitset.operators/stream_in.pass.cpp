/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is, bitset<N>& x);

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
  {
    cuda::std::istringstream in("01011010");
    cuda::std::bitset<8> b;
    in >> b;
    assert(b.to_ulong() == 0x5A);
  }
  {
    // Make sure that input-streaming an empty bitset does not cause the
    // failbit to be set (LWG 3199).
    cuda::std::istringstream in("01011010");
    cuda::std::bitset<0> b;
    in >> b;
    assert(b.to_string() == "");
    assert(!in.bad());
    assert(!in.fail());
    assert(!in.eof());
    assert(in.good());
  }
#  if TEST_HAS_EXCEPTIONS()
  {
    cuda::std::stringbuf sb;
    cuda::std::istream is(&sb);
    is.exceptions(cuda::std::ios::failbit);

    bool threw = false;
    try
    {
      cuda::std::bitset<8> b;
      is >> b;
    }
    catch (cuda::std::ios::failure const&)
    {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
  }
  {
    cuda::std::stringbuf sb;
    cuda::std::istream is(&sb);
    is.exceptions(cuda::std::ios::eofbit);

    bool threw = false;
    try
    {
      cuda::std::bitset<8> b;
      is >> b;
    }
    catch (cuda::std::ios::failure const&)
    {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
  }
#  endif // TEST_HAS_EXCEPTIONS()

  return 0;
}

#endif
