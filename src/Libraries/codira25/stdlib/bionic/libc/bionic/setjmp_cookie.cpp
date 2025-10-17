/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/cdefs.h>

#include <async_safe/log.h>

#include "private/bionic_arc4random.h"
#include "private/bionic_globals.h"

void __libc_init_setjmp_cookie(libc_globals* globals) {
  long value;
  __libc_safe_arc4random_buf(&value, sizeof(value));

  // Mask off the last bit to store the signal flag.
  globals->setjmp_cookie = value & ~1;
}

extern "C" __LIBC_HIDDEN__ long __bionic_setjmp_cookie_get(long sigflag) {
  if (sigflag & ~1) {
    async_safe_fatal("unexpected sigflag value: %ld", sigflag);
  }

  return __libc_globals->setjmp_cookie | sigflag;
}

// Aborts if cookie doesn't match, returns the signal flag otherwise.
extern "C" __LIBC_HIDDEN__ long __bionic_setjmp_cookie_check(long cookie) {
  if (__libc_globals->setjmp_cookie != (cookie & ~1)) {
    async_safe_fatal("setjmp cookie mismatch");
  }

  return cookie & 1;
}

extern "C" __LIBC_HIDDEN__ long __bionic_setjmp_checksum_mismatch() {
  async_safe_fatal("setjmp checksum mismatch");
}
