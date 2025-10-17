/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#include "internal.h"

#if !defined(OPENSSL_THREADS)

void CRYPTO_MUTEX_init(CRYPTO_MUTEX *lock) {}

void CRYPTO_MUTEX_lock_read(CRYPTO_MUTEX *lock) {}

void CRYPTO_MUTEX_lock_write(CRYPTO_MUTEX *lock) {}

void CRYPTO_MUTEX_unlock_read(CRYPTO_MUTEX *lock) {}

void CRYPTO_MUTEX_unlock_write(CRYPTO_MUTEX *lock) {}

void CRYPTO_MUTEX_cleanup(CRYPTO_MUTEX *lock) {}

void CRYPTO_once(CRYPTO_once_t *once, void (*init)(void)) {
  if (*once) {
    return;
  }
  *once = 1;
  init();
}

static void *g_thread_locals[NUM_OPENSSL_THREAD_LOCALS];

void *CRYPTO_get_thread_local(thread_local_data_t index) {
  return g_thread_locals[index];
}

int CRYPTO_set_thread_local(thread_local_data_t index, void *value,
                            thread_local_destructor_t destructor) {
  g_thread_locals[index] = value;
  return 1;
}

#endif  // !OPENSSL_THREADS
