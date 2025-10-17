/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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

//===--- Libc.h - Imports from the C library --------------------*- C++ -*-===//
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
//
// Extra utilities for libc
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BACKTRACING_LIBC_H
#define LANGUAGE_BACKTRACING_LIBC_H

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>

// .. Codira affordances ........................................................

#ifdef __cplusplus
namespace language {
namespace runtime {
namespace backtrace {
#endif

/* open() is usually declared as a variadic function; these don't import into
   Codira. */
static inline int _language_open(const char *filename, int oflag, int mode) {
  return open(filename, oflag, mode);
}

/* errno is typically not going to be easily accessible (it's often a macro),
   so add a get_errno() function to do that. */
static inline int _language_get_errno() { return errno; }
static void _language_set_errno(int err) { errno = err; }

#ifdef __cplusplus
} // namespace backtrace
} // namespace runtime
} // namespace language
#endif

#endif // LANGUAGE_BACKTRACING_LIBC_H
