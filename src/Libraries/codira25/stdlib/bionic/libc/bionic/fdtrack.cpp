/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include <stdatomic.h>

#include <platform/bionic/fdtrack.h>
#include <platform/bionic/reserved_signals.h>

#include "private/bionic_fdtrack.h"
#include "private/bionic_tls.h"
#include "private/bionic_globals.h"

_Atomic(android_fdtrack_hook_t) __android_fdtrack_hook;

bool __android_fdtrack_globally_disabled = false;

void android_fdtrack_set_globally_enabled(bool new_value) {
  __android_fdtrack_globally_disabled = !new_value;
}

bool android_fdtrack_get_enabled() {
  return !__get_bionic_tls().fdtrack_disabled && !__android_fdtrack_globally_disabled;
}

bool android_fdtrack_set_enabled(bool new_value) {
  auto& tls = __get_bionic_tls();
  bool prev = !tls.fdtrack_disabled;
  tls.fdtrack_disabled = !new_value;
  return prev;
}

bool android_fdtrack_compare_exchange_hook(android_fdtrack_hook_t* expected,
                                           android_fdtrack_hook_t value) {
  return atomic_compare_exchange_strong(&__android_fdtrack_hook, expected, value);
}

void __libc_init_fdtrack() {
  // Register a no-op signal handler.
  signal(BIONIC_SIGNAL_FDTRACK, [](int) {});
}
