/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
#include <sys/sem.h>

#include <stdarg.h>
#include <sys/syscall.h>
#include <unistd.h>

int semctl(int id, int num, int cmd, ...) {
#if !defined(__LP64__)
  // Annoyingly, the kernel requires this for 32-bit but rejects it for 64-bit.
  cmd |= IPC_64;
#endif
  va_list ap;
  va_start(ap, cmd);
  semun arg = va_arg(ap, semun);
  va_end(ap);
  return syscall(SYS_semctl, id, num, cmd, arg);
}

int semget(key_t key, int n, int flags) {
  return syscall(SYS_semget, key, n, flags);
}

int semop(int id, sembuf* ops, size_t op_count) {
  return semtimedop(id, ops, op_count, nullptr);
}

int semtimedop(int id, sembuf* ops, size_t op_count, const timespec* ts) {
#if defined(SYS_semtimedop)
  return syscall(SYS_semtimedop, id, ops, op_count, ts);
#else
  // 32-bit x86 -- the only architecture without semtimedop(2) -- only has
  // semtimedop_time64(2), but since we don't have any timespec64 stuff,
  // it's less painful for us to just stick with the legacy ipc(2) here.
  return syscall(SYS_ipc, SEMTIMEDOP, id, op_count, 0, ops, ts);
#endif
}
