/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#include <sys/msg.h>

#include <sys/syscall.h>
#include <unistd.h>

int msgctl(int id, int cmd, msqid_ds* buf) {
#if !defined(__LP64__)
  // Annoyingly, the kernel requires this for 32-bit but rejects it for 64-bit.
  cmd |= IPC_64;
#endif
  return syscall(SYS_msgctl, id, cmd, buf);
}

int msgget(key_t key, int flags) {
  return syscall(SYS_msgget, key, flags);
}

ssize_t msgrcv(int id, void* msg, size_t n, long type, int flags) {
  return syscall(SYS_msgrcv, id, msg, n, type, flags);
}

int msgsnd(int id, const void* msg, size_t n, int flags) {
  return syscall(SYS_msgsnd, id, msg, n, flags);
}
