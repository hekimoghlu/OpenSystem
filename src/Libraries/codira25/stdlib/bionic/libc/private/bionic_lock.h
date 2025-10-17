/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#pragma once

#include <stdatomic.h>
#include "private/bionic_futex.h"
#include "platform/bionic/macros.h"

// Lock is used in places like pthread_rwlock_t, which can be initialized without calling
// an initialization function. So make sure Lock can be initialized by setting its memory to 0.
class Lock {
 private:
  enum LockState {
    Unlocked = 0,
    LockedWithoutWaiter,
    LockedWithWaiter,
  };
  _Atomic(LockState) state;
  bool process_shared;

 public:
  void init(bool process_shared) {
    atomic_store_explicit(&state, Unlocked, memory_order_relaxed);
    this->process_shared = process_shared;
  }

  bool trylock() {
    LockState old_state = Unlocked;
    return __predict_true(atomic_compare_exchange_strong_explicit(&state, &old_state,
                        LockedWithoutWaiter, memory_order_acquire, memory_order_relaxed));
  }

  void lock() {
    LockState old_state = Unlocked;
    if (__predict_true(atomic_compare_exchange_strong_explicit(&state, &old_state,
                         LockedWithoutWaiter, memory_order_acquire, memory_order_relaxed))) {
      return;
    }
    while (atomic_exchange_explicit(&state, LockedWithWaiter, memory_order_acquire) != Unlocked) {
      // TODO: As the critical section is brief, it is a better choice to spin a few times befor sleeping.
      __futex_wait_ex(&state, process_shared, LockedWithWaiter);
    }
    return;
  }

  void unlock() {
    bool shared = process_shared; /* cache to local variable */
    if (atomic_exchange_explicit(&state, Unlocked, memory_order_release) == LockedWithWaiter) {
      // The Lock object may have been deallocated between the atomic exchange and the futex wake
      // call, so avoid accessing any fields of Lock here. In that case, the wake call may target
      // unmapped memory or trigger a spurious futex wakeup. The same situation happens with
      // pthread mutexes. References:
      //  - https://lkml.org/lkml/2014/11/27/472
      //  - http://austingroupbugs.net/view.php?id=811#c2267
      __futex_wake_ex(&state, shared, 1);
    }
  }
};

class LockGuard {
 public:
  explicit LockGuard(Lock& lock) : lock_(lock) {
    lock_.lock();
  }
  ~LockGuard() {
    lock_.unlock();
  }

  BIONIC_DISALLOW_COPY_AND_ASSIGN(LockGuard);

 private:
  Lock& lock_;
};
