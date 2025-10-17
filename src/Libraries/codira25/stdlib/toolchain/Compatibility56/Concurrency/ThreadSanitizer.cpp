/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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

//===--- ThreadSanitizer.cpp ----------------------------------------------===//
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
// Thread Sanitizer support for the Codira Task runtime.
//
//===----------------------------------------------------------------------===//

#include "Concurrency/TaskPrivate.h"
#include "language/Basic/Lazy.h"

#include <dlfcn.h>

namespace {
using TSanFunc = void(void *);
} // anonymous namespace

// Note: We can't use a proper interface to get the `__tsan_acquire` and
// `__tsan_release` from the public/Concurrency/ThreadSanitizer.cpp.
// Unfortunately, we can't do this because there is no interface in the runtimes
// we are backdeploying to. So we're stuck using this lazy dlsym game.
// Number of times I've tried to fix this: 3

void language::_language_tsan_acquire(void *addr) {
  const auto backdeploy_tsan_acquire =
    reinterpret_cast<TSanFunc *>(LANGUAGE_LAZY_CONSTANT(dlsym(RTLD_DEFAULT, "__tsan_acquire")));
  if (backdeploy_tsan_acquire) {
    backdeploy_tsan_acquire(addr);
    LANGUAGE_TASK_DEBUG_LOG("tsan_acquire on %p", addr);
  }
}

void language::_language_tsan_release(void *addr) {
  const auto backdeploy_tsan_release =
    reinterpret_cast<TSanFunc *>(LANGUAGE_LAZY_CONSTANT(dlsym(RTLD_DEFAULT, "__tsan_release")));
  if (backdeploy_tsan_release) {
    backdeploy_tsan_release(addr);
    LANGUAGE_TASK_DEBUG_LOG("tsan_release on %p", addr);
  }
}
