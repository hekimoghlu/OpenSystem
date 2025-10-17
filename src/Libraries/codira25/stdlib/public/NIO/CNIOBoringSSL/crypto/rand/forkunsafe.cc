/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include <CNIOBoringSSL_rand.h>

#include <stdlib.h>

#include "../fipsmodule/rand/internal.h"
#include "../internal.h"


// g_buffering_enabled is one if fork-unsafe buffering has been enabled and zero
// otherwise.
static CRYPTO_atomic_u32 g_buffering_enabled;

#if !defined(OPENSSL_WINDOWS)
void RAND_enable_fork_unsafe_buffering(int fd) {
  // We no longer support setting the file-descriptor with this function.
  if (fd != -1) {
    abort();
  }

  CRYPTO_atomic_store_u32(&g_buffering_enabled, 1);
}

void RAND_disable_fork_unsafe_buffering(void) {
  CRYPTO_atomic_store_u32(&g_buffering_enabled, 0);
}
#endif

int rand_fork_unsafe_buffering_enabled(void) {
  return CRYPTO_atomic_load_u32(&g_buffering_enabled) != 0;
}
