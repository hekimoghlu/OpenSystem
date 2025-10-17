/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#include "_AtomicsShims.h"

// FIXME: These should be static inline header-only shims, but Codira 5.3 doesn't
// like calls to language_retain_n/language_release_n appearing in Codira code, not
// even when imported through C. (See https://bugs.code.org/browse/SR-13708)

#if defined(__APPLE__) && defined(__MACH__)
// Ensure we link with liblanguageCore.dylib even when the build system decides
// to build this module as a standalone library.
// (See https://github.com/apple/language-atomics/issues/55)
__asm__(".linker_option \"-llanguageCore\"\n");
#endif

void _sa_retain_n(void *object, uint32_t n)
{
  extern void *language_retain_n(void *object, uint32_t n);
  language_retain_n(object, n);
}

void _sa_release_n(void *object, uint32_t n)
{
  extern void language_release_n(void *object, uint32_t n);
  language_release_n(object, n);
}
