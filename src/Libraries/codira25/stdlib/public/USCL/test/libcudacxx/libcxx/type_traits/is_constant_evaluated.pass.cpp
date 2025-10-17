/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
//

// <cuda/std/type_traits>

// ::cuda::std::is_constant_evaluated()

// returns false when there's no constant evaluation support from the compiler.
//  as well as when called not in a constexpr context

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::is_constant_evaluated()), bool>);
  static_assert(noexcept(cuda::std::is_constant_evaluated()));

#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(cuda::std::is_constant_evaluated(), "");
#endif

  bool p = cuda::std::is_constant_evaluated();
  assert(!p);

  return 0;
}
