/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

//===--- BacktraceUtils.cpp - Private backtracing utilities -----*- C++ -*-===//
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
// Defines some async-signal-safe formatting functions.
//
//===----------------------------------------------------------------------===//

#include "BacktracePrivate.h"

namespace language {
namespace runtime {
namespace backtracing {

// We can't call sprintf() here because we're in a signal handler,
// so we need to be async-signal-safe.
LANGUAGE_RUNTIME_STDLIB_INTERNAL void
_language_formatAddress(uintptr_t addr, char buffer[18])
{
  char *ptr = buffer + 18;
  *--ptr = '\0';
  while (ptr > buffer) {
    char digit = '0' + (addr & 0xf);
    if (digit > '9')
      digit += 'a' - '0' - 10;
    *--ptr = digit;
    addr >>= 4;
    if (!addr)
      break;
  }

  // Left-justify in the buffer
  if (ptr > buffer) {
    char *pt2 = buffer;
    while (*ptr)
      *pt2++ = *ptr++;
    *pt2++ = '\0';
  }
}

// See above; we can't use sprintf() here.
LANGUAGE_RUNTIME_STDLIB_INTERNAL void
_language_formatUnsigned(unsigned u, char buffer[22])
{
  char *ptr = buffer + 22;
  *--ptr = '\0';
  while (ptr > buffer) {
    char digit = '0' + (u % 10);
    *--ptr = digit;
    u /= 10;
    if (!u)
      break;
  }

  // Left-justify in the buffer
  if (ptr > buffer) {
    char *pt2 = buffer;
    while (*ptr)
      *pt2++ = *ptr++;
    *pt2++ = '\0';
  }
}

} // namespace backtracing
} // namespace runtime
} // namespace language
