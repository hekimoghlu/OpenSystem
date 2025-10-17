/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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

// <cuda/std/type_traits>

// constexpr bool is_constant_evaluated() noexcept; // C++20

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

#if TEST_STD_VER > 2017
#  ifndef __cccl_lib_is_constant_evaluated
#    if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#      error __cccl_lib_is_constant_evaluated should be defined
#    endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
#  endif // __cccl_lib_is_constant_evaluated
#endif // TEST_STD_VER > 2017

template <bool>
struct InTemplate
{};

int main(int, char**)
{
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  // Test the signature
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::is_constant_evaluated()), bool>);
    static_assert(noexcept(cuda::std::is_constant_evaluated()));
    constexpr bool p = cuda::std::is_constant_evaluated();
    assert(p);
  }
  // Test the return value of the builtin for basic sanity only. It's the
  // compilers job to test tho builtin for correctness.
  {
    static_assert(cuda::std::is_constant_evaluated(), "");
    bool p = cuda::std::is_constant_evaluated();
    assert(!p);
    static_assert(cuda::std::is_same_v<InTemplate<cuda::std::is_constant_evaluated()>, InTemplate<true>>);
    static int local_static = cuda::std::is_constant_evaluated() ? 42 : -1;
    assert(local_static == 42);
  }
#endif
  return 0;
}
