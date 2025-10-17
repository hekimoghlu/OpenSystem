/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

//===--- AsyncStream.cpp - Multi-resume locking interface -----------------===//
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

#include <new>

#include "language/Runtime/Config.h"
#include "language/Threading/Mutex.h"

namespace language {
// return the size in words for the given mutex primitive
LANGUAGE_CC(language)
extern "C"
size_t _language_async_stream_lock_size() {
  size_t words = sizeof(Mutex) / sizeof(void *);
  if (words < 1) { return 1; }
  return words;
}

LANGUAGE_CC(language)
extern "C" void _language_async_stream_lock_init(Mutex &lock) {
  new (&lock) Mutex();
}

LANGUAGE_CC(language)
extern "C" void _language_async_stream_lock_lock(Mutex &lock) { lock.lock(); }

LANGUAGE_CC(language)
extern "C" void _language_async_stream_lock_unlock(Mutex &lock) { lock.unlock(); }
}
