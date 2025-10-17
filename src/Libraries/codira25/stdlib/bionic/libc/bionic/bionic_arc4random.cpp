/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#include "private/bionic_arc4random.h"

#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <unistd.h>

#include <async_safe/log.h>

void __libc_safe_arc4random_buf(void* buf, size_t n) {
  // Only call arc4random_buf once we have `/dev/urandom` because getentropy(3)
  // will fall back to using `/dev/urandom` if getrandom(2) fails, and abort if
  // if can't use `/dev/urandom`.
  static bool have_urandom = access("/dev/urandom", R_OK) == 0;
  if (have_urandom) {
    arc4random_buf(buf, n);
    return;
  }

  static size_t at_random_bytes_consumed = 0;
  if (at_random_bytes_consumed + n > 16) {
    async_safe_fatal("ran out of AT_RANDOM bytes, have %zu, requested %zu",
                     16 - at_random_bytes_consumed, n);
  }

  memcpy(buf, reinterpret_cast<char*>(getauxval(AT_RANDOM)) + at_random_bytes_consumed, n);
  at_random_bytes_consumed += n;
  return;
}
