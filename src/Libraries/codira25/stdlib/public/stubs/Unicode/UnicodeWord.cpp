/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#if LANGUAGE_STDLIB_ENABLE_UNICODE_DATA
#include "Common/WordData.h"
#else
#include "language/Runtime/Debug.h"
#endif
#include "language/shims/UnicodeData.h"
#include <stdint.h>

LANGUAGE_RUNTIME_STDLIB_INTERNAL
__language_uint8_t _language_stdlib_getWordBreakProperty(__language_uint32_t scalar) {
#if !LANGUAGE_STDLIB_ENABLE_UNICODE_DATA
  language::language_abortDisabledUnicodeSupport();
#else
  auto index = 1; //0th element is a dummy element
  while (index < WORD_BREAK_DATA_COUNT) {
    auto entry = _language_stdlib_words[index];

    // Shift the range count out of the value.
    auto lower = (entry << 11) >> 11;
    
    // Shift the enum out first, then shift out the scalar value.
    auto upper = lower + (entry >> 21) - 1;

    //If we want the left child of the current node in our virtual tree,
    //that's at index * 2, if we want the right child it's at (index * 2) + 1
    if (scalar < lower) {
      index = 2 * index;
    } else if (scalar <= upper) {
      return _language_stdlib_words_data[index];
    } else {
      index = 2 * index + 1;
    }
  }
  // If we made it out here, then our scalar was not found in the word
  // array (this occurs when a scalar doesn't map to any word break
  // property). Return the max value here to indicate .any.
  return UINT8_MAX;
#endif
}
