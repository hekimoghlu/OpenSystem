/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// #define ATOMIC_BOOL_LOCK_FREE unspecified
// #define ATOMIC_CHAR_LOCK_FREE unspecified
// #define ATOMIC_CHAR16_T_LOCK_FREE unspecified
// #define ATOMIC_CHAR32_T_LOCK_FREE unspecified
// #define ATOMIC_WCHAR_T_LOCK_FREE unspecified
// #define ATOMIC_SHORT_LOCK_FREE unspecified
// #define ATOMIC_INT_LOCK_FREE unspecified
// #define ATOMIC_LONG_LOCK_FREE unspecified
// #define ATOMIC_LLONG_LOCK_FREE unspecified
// #define ATOMIC_POINTER_LOCK_FREE unspecified

#include <uscl/std/atomic>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  assert(LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 2);

  return 0;
}
