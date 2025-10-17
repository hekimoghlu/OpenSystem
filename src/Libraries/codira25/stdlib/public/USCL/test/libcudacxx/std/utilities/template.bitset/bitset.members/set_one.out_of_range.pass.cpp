/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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

// UNSUPPORTED: libcpp-no-exceptions

// bitset<N>& set(size_t pos, bool val = true); // constexpr since C++23

// Make sure we throw ::std::out_of_range when calling set() on an OOB index.

#include <uscl/std/bitset>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(
    NV_IS_HOST,
    {
      cuda::std::bitset<0> v;
      try
      {
        v.set(0);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    } {
      cuda::std::bitset<1> v("0");
      try
      {
        v.set(2);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    } {
      cuda::std::bitset<10> v("0000000000");
      try
      {
        v.set(10);
        assert(false);
      }
      catch (::std::out_of_range const&)
      {}
    })
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
