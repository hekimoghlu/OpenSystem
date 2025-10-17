/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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

// enum class endian;
// <cuda/std/bit>

#include <uscl/std/bit>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/cstring>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_scoped_enum_v<cuda::std::endian>);

  // test that the enumeration values exist
  static_assert(cuda::std::endian::little == cuda::std::endian::little);
  static_assert(cuda::std::endian::big == cuda::std::endian::big);
  static_assert(cuda::std::endian::native == cuda::std::endian::native);
  static_assert(cuda::std::endian::little != cuda::std::endian::big);

  //  Try to check at runtime
  {
    cuda::std::uint32_t i = 0x01020304;
    char c[4];
    static_assert(sizeof(i) == sizeof(c));

    cuda::std::memcpy(c, &i, sizeof(c));
    assert((c[0] == 1) == (cuda::std::endian::native == cuda::std::endian::big));
  }

  return 0;
}
