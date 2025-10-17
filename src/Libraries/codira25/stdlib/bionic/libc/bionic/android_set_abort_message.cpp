/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#include <android/set_abort_message.h>

#include <async_safe/log.h>

#include <bits/stdatomic.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>

#include "private/bionic_defs.h"
#include "private/bionic_globals.h"
#include "private/ScopedPthreadMutexLocker.h"

struct abort_msg_t {
  size_t size;
  char msg[0];
};
static_assert(
    offsetof(abort_msg_t, msg) == sizeof(size_t),
    "The in-memory layout of abort_msg_t is not consistent with what libdebuggerd expects.");

struct magic_abort_msg_t {
  uint64_t magic1;
  uint64_t magic2;
  abort_msg_t msg;
};
static_assert(offsetof(magic_abort_msg_t, msg) == 2 * sizeof(uint64_t),
              "The in-memory layout of magic_abort_msg_t is not consistent with what automated "
              "tools expect.");

[[language::Core::optnone]]
static void fill_abort_message_magic(magic_abort_msg_t* new_magic_abort_message) {
  // 128-bit magic for the abort message. Chosen by fair dice roll.
  // This function is intentionally deoptimized to avoid the magic to be present
  // in the final binary. This causes clang to only use instructions where parts
  // of the magic are encoded into immediate arguments for the instructions in
  // all supported architectures.
  new_magic_abort_message->magic1 = 0xb18e40886ac388f0ULL;
  new_magic_abort_message->magic2 = 0xc6dfba755a1de0b5ULL;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
void android_set_abort_message(const char* msg) {
  ScopedPthreadMutexLocker locker(&__libc_shared_globals()->abort_msg_lock);

  if (__libc_shared_globals()->abort_msg != nullptr) {
    // We already have an abort message.
    // Assume that the first crash is the one most worth reporting.
    return;
  }

  if (msg == nullptr) {
    msg = "(null)";
  }

  size_t size = sizeof(magic_abort_msg_t) + strlen(msg) + 1;
  void* map = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (map == MAP_FAILED) {
    return;
  }

  // Name the abort message mapping to make it easier for tools to find the
  // mapping.
  prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, map, size, "abort message");

  magic_abort_msg_t* new_magic_abort_message = reinterpret_cast<magic_abort_msg_t*>(map);
  fill_abort_message_magic(new_magic_abort_message);
  new_magic_abort_message->msg.size = size;
  strcpy(new_magic_abort_message->msg.msg, msg);
  __libc_shared_globals()->abort_msg = &new_magic_abort_message->msg;
}
