/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

//===--- Malloc.h - Aligned malloc interface --------------------*- C++ -*-===//
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
//  This file provides an implementation of C11 aligned_alloc(3) for platforms
//  that don't have it yet, using posix_memalign(3).
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_MALLOC_H
#define LANGUAGE_BASIC_MALLOC_H

#include <cassert>
#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace language {

inline void *AlignedAlloc(size_t size, size_t align) {
#if defined(_WIN32)
  void *r = _aligned_malloc(size, align);
  assert(r && "_aligned_malloc failed");
#elif __STDC_VERSION__-0 >= 201112l
  // C11 supports aligned_alloc
  void *r = aligned_alloc(align, size);
  assert(r && "aligned_alloc failed");
#else
  // posix_memalign only accepts alignments greater than sizeof(void*).
  if (align < sizeof(void *))
    align = sizeof(void *);

  void *r = nullptr;
  int res = posix_memalign(&r, align, size);
  assert(res == 0 && "posix_memalign failed");
  (void)res; // Silence the unused variable warning.
#endif
  return r;
}

inline void AlignedFree(void *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

} // end namespace language

#endif // LANGUAGE_BASIC_MALLOC_H
