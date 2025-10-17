/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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

// test climits

#include <uscl/std/climits>

#include "test_macros.h"

#ifndef CHAR_BIT
#  error CHAR_BIT not defined
#endif

#ifndef SCHAR_MIN
#  error SCHAR_MIN not defined
#endif

#ifndef SCHAR_MAX
#  error SCHAR_MAX not defined
#endif

#ifndef UCHAR_MAX
#  error UCHAR_MAX not defined
#endif

#ifndef CHAR_MIN
#  error CHAR_MIN not defined
#endif

#ifndef CHAR_MAX
#  error CHAR_MAX not defined
#endif

// #ifndef MB_LEN_MAX
// #error MB_LEN_MAX not defined
// #endif

#ifndef SHRT_MIN
#  error SHRT_MIN not defined
#endif

#ifndef SHRT_MAX
#  error SHRT_MAX not defined
#endif

#ifndef USHRT_MAX
#  error USHRT_MAX not defined
#endif

#ifndef INT_MIN
#  error INT_MIN not defined
#endif

#ifndef INT_MAX
#  error INT_MAX not defined
#endif

#ifndef UINT_MAX
#  error UINT_MAX not defined
#endif

#ifndef LONG_MIN
#  error LONG_MIN not defined
#endif

#ifndef LONG_MAX
#  error LONG_MAX not defined
#endif

#ifndef ULONG_MAX
#  error ULONG_MAX not defined
#endif

#ifndef LLONG_MIN
#  error LLONG_MIN not defined
#endif

#ifndef LLONG_MAX
#  error LLONG_MAX not defined
#endif

#ifndef ULLONG_MAX
#  error ULLONG_MAX not defined
#endif

// test if _CCCL_CHAR_IS_UNSIGNED() detection for NVRTC works correctly
// if not, go take a look at cuda::std::is_unsigned_v
#if TEST_COMPILER(NVRTC)
#  include <cuda/std/type_traits>
static_assert(_CCCL_CHAR_IS_UNSIGNED() == cuda::std::is_unsigned_v<char>, "");
#endif

int main(int, char**)
{
  return 0;
}
