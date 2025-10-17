/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

//==--- Linux.cpp - Threading abstraction implementation ------- -*-C++ -*-===//
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
// Implements threading support for Linux
//
//===----------------------------------------------------------------------===//

#if LANGUAGE_THREADING_LINUX

#include "language/Threading/Impl.h"
#include "language/Threading/Errors.h"
#include "language/Threading/ThreadSanitizer.h"

namespace {

#pragma clang diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"

class MainThreadRememberer {
private:
  pthread_t mainThread_;

public:
  MainThreadRememberer() { mainThread_ = pthread_self(); }

  pthread_t main_thread() const { return mainThread_; }
};

MainThreadRememberer rememberer;

#if !defined(__LP64__) && !defined(_LP64)
pthread_mutex_t once_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

#pragma clang diagnostic pop

} // namespace

using namespace language;
using namespace threading_impl;

bool language::threading_impl::thread_is_main() {
  return pthread_equal(pthread_self(), rememberer.main_thread());
}

void language::threading_impl::once_slow(once_t &predicate, void (*fn)(void *),
                                      void *context) {
  // On 32-bit Linux we can't have per-once locks
#if defined(__LP64__) || defined(_LP64)
  linux::ulock_lock(&predicate.lock);
#else
  pthread_mutex_lock(&once_mutex);
#endif
  if (predicate.flag.load(std::memory_order_acquire) == 0) {
    fn(context);
    predicate.flag.store(tsan::enabled() ? 1 : -1, std::memory_order_release);
  }
#if defined(__LP64__) || defined(_LP64)
  linux::ulock_unlock(&predicate.lock);
#else
  pthread_mutex_unlock(&once_mutex);
#endif
}

std::optional<language::threading_impl::stack_bounds>
language::threading_impl::thread_get_current_stack_bounds() {
  pthread_attr_t attr;
  size_t size = 0;
  void *begin = nullptr;

  if (!pthread_getattr_np(pthread_self(), &attr)) {
    if (!pthread_attr_getstack(&attr, &begin, &size)) {
      stack_bounds result = { begin, (char *)begin + size };

      pthread_attr_destroy(&attr);

      return result;
    }

    pthread_attr_destroy(&attr);
  }

  return {};
}

#endif // LANGUAGE_THREADING_LINUX
