/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

//===--- LibcShims.cpp ----------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)
#define _REENTRANT
#include <math.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <io.h>
#include <stdlib.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include "language/shims/LibcShims.h"

#if defined(_WIN32)
static void __attribute__((__constructor__))
_language_stdlib_configure_console_mode(void) {
  static UINT uiPrevConsoleCP = GetConsoleOutputCP();
  atexit([]() { SetConsoleOutputCP(uiPrevConsoleCP); });
  SetConsoleOutputCP(CP_UTF8);
}
#endif

LANGUAGE_RUNTIME_STDLIB_INTERNAL
__language_size_t _language_stdlib_fwrite_stdout(const void *ptr,
                                           __language_size_t size,
                                           __language_size_t nitems) {
  return fwrite(ptr, size, nitems, stdout);
}
