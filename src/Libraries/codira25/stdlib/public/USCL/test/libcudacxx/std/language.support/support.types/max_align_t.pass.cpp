/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

#include <uscl/std/cstddef>
#include <uscl/std/type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type
#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <stdio.h>
#endif // TEST_COMPILER(NVRTC)

int main(int, char**)
{
#if TEST_STD_VER > 2017
  //  P0767
  static_assert(cuda::std::is_trivial<cuda::std::max_align_t>::value,
                "cuda::std::is_trivial<cuda::std::max_align_t>::value");
  static_assert(cuda::std::is_standard_layout<cuda::std::max_align_t>::value,
                "cuda::std::is_standard_layout<cuda::std::max_align_t>::value");
#else
  static_assert(cuda::std::is_pod<cuda::std::max_align_t>::value, "cuda::std::is_pod<cuda::std::max_align_t>::value");
#endif
  static_assert((cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long long>::value),
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long long>::value");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long double>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long double>::value");
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<void*>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<void*>::value");

  return 0;
}
