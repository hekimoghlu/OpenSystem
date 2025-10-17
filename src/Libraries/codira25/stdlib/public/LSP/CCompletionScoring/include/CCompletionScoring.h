/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#ifndef SOURCEKITLSP_CCOMPLETIONSCORING_H
#define SOURCEKITLSP_CCOMPLETIONSCORING_H

#define _GNU_SOURCE
#include <string.h>

static inline void *sourcekitlsp_memmem(const void *haystack, size_t haystack_len, const void *needle, size_t needle_len) {
  #if defined(_WIN32) && !defined(__CYGWIN__)
  // memmem is not available on Windows
  if (!haystack || haystack_len == 0) {
    return NULL;
  }
  if (!needle || needle_len == 0) {
    return NULL;
  }
  if (needle_len > haystack_len) {
    return NULL;
  }

  for (size_t offset = 0; offset <= haystack_len - needle_len; ++offset) {
    if (memcmp(haystack + offset, needle, needle_len) == 0) {
      return (void *)haystack + offset;
    }
  }
  return NULL;
  #else
  return memmem(haystack, haystack_len, needle, needle_len);
  #endif
}

#endif // SOURCEKITLSP_CCOMPLETIONSCORING_H
