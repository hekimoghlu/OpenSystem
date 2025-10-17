/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#include <pthread.h>
#include <stdatomic.h>

#include "private/bionic_futex.h"

#define ONCE_INITIALIZATION_NOT_YET_STARTED   0
#define ONCE_INITIALIZATION_UNDERWAY          1
#define ONCE_INITIALIZATION_COMPLETE          2

/* NOTE: this implementation doesn't support a init function that throws a C++ exception
 *       or calls fork()
 */
int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  static_assert(sizeof(atomic_int) == sizeof(pthread_once_t),
                "pthread_once_t should actually be atomic_int in implementation.");

  // We prefer casting to atomic_int instead of declaring pthread_once_t to be atomic_int directly.
  // Because using the second method pollutes pthread.h, and causes an error when compiling libcxx.
  atomic_int* once_control_ptr = reinterpret_cast<atomic_int*>(once_control);

  // First check if the once is already initialized. This will be the common
  // case and we want to make this as fast as possible. Note that this still
  // requires a load_acquire operation here to ensure that all the
  // stores performed by the initialization function are observable on
  // this CPU after we exit.
  int old_value = atomic_load_explicit(once_control_ptr, memory_order_acquire);

  while (true) {
    if (__predict_true(old_value == ONCE_INITIALIZATION_COMPLETE)) {
      return 0;
    }

    // Try to atomically set the initialization underway flag. This requires a compare exchange
    // in a loop, and we may need to exit prematurely if the initialization is complete.
    if (!atomic_compare_exchange_weak_explicit(once_control_ptr, &old_value,
                                               ONCE_INITIALIZATION_UNDERWAY,
                                               memory_order_acquire, memory_order_acquire)) {
      continue;
    }

    if (old_value == ONCE_INITIALIZATION_NOT_YET_STARTED) {
      // We got here first, we can handle the initialization.
      (*init_routine)();

      // Do a store_release indicating that initialization is complete.
      atomic_store_explicit(once_control_ptr, ONCE_INITIALIZATION_COMPLETE, memory_order_release);

      // Wake up any waiters, if any.
      __futex_wake_ex(once_control_ptr, 0, INT_MAX);
      return 0;
    }

    // The initialization is underway, wait for its finish.
    __futex_wait_ex(once_control_ptr, 0, old_value, false, nullptr);
    old_value = atomic_load_explicit(once_control_ptr, memory_order_acquire);
  }
}
