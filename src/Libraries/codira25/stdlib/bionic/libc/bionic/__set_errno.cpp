/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#include <errno.h>

// This function is called from our assembler syscall stubs.
// C/C++ code should just assign 'errno' instead.

// The return type is 'long' because we use the same routine in calls
// that return an int as in ones that return a ssize_t. On a 32-bit
// system these are the same size, but on a 64-bit system they're not.
// 'long' gives us 32-bit on 32-bit systems, 64-bit on 64-bit systems.

// Since __set_errno was mistakenly exposed in <errno.h> in the 32-bit
// NDK, use a differently named internal function for the system call
// stubs. This avoids having the stubs .hidden directives accidentally
// hide __set_errno for old NDK apps.

// This one is for internal use only and used by both LP32 and LP64 assembler.
extern "C" __LIBC_HIDDEN__ long __set_errno_internal(int n) {
  errno = n;
  return -1;
}
