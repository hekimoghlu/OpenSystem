/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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

#include "private/bionic_lock.h"

// User-level spinlocks can be hazardous to battery life on Android.
// We implement a simple compromise that behaves mostly like a spinlock,
// but prevents excessively long spinning.

struct pthread_spinlock_internal_t {
  Lock lock;
};

static_assert(sizeof(pthread_spinlock_t) == sizeof(pthread_spinlock_internal_t),
              "pthread_spinlock_t should actually be pthread_spinlock_internal_t.");

static_assert(alignof(pthread_spinlock_t) >= 4,
              "pthread_spinlock_t should fulfill the alignment of pthread_spinlock_internal_t.");

static inline pthread_spinlock_internal_t* __get_internal_spinlock(pthread_spinlock_t* lock) {
  return reinterpret_cast<pthread_spinlock_internal_t*>(lock);
}

int pthread_spin_init(pthread_spinlock_t* lock_interface, int pshared) {
  pthread_spinlock_internal_t* lock = __get_internal_spinlock(lock_interface);
  lock->lock.init(pshared);
  return 0;
}

int pthread_spin_destroy(pthread_spinlock_t* lock_interface) {
  pthread_spinlock_internal_t* lock = __get_internal_spinlock(lock_interface);
  return lock->lock.trylock() ? 0 : EBUSY;
}

int pthread_spin_trylock(pthread_spinlock_t* lock_interface) {
  pthread_spinlock_internal_t* lock = __get_internal_spinlock(lock_interface);
  return lock->lock.trylock() ? 0 : EBUSY;
}

int pthread_spin_lock(pthread_spinlock_t* lock_interface) {
  pthread_spinlock_internal_t* lock = __get_internal_spinlock(lock_interface);
  for (int i = 0; i < 10000; ++i) {
    if (lock->lock.trylock()) {
      return 0;
    }
  }
  lock->lock.lock();
  return 0;
}

int pthread_spin_unlock(pthread_spinlock_t* lock_interface) {
  pthread_spinlock_internal_t* lock = __get_internal_spinlock(lock_interface);
  lock->lock.unlock();
  return 0;
}
