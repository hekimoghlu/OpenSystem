/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

//===--- Error.cpp - Error handling support code --------------------------===//
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

#include "language/Threading/Errors.h"
#include <cstdio>

#include "Error.h"

// language::fatalError is not exported from liblanguageCore and not shared, so define another
// internal function instead.
LANGUAGE_NORETURN
LANGUAGE_VFORMAT(2)
void language::language_Concurrency_fatalErrorv(uint32_t flags, const char *format,
                                          va_list val) {
#if !LANGUAGE_CONCURRENCY_EMBEDDED
  vfprintf(stderr, format, val);
#else
  vprintf(format, val);
#endif
  abort();
}

LANGUAGE_NORETURN
LANGUAGE_FORMAT(2, 3)
void language::language_Concurrency_fatalError(uint32_t flags, const char *format,
                                         ...) {
  va_list val;

  va_start(val, format);
  language_Concurrency_fatalErrorv(flags, format, val);
}
