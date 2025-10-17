/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#ifndef _UAPI_LINUX_IPC_H
#define _UAPI_LINUX_IPC_H
#include <linux/types.h>
#define IPC_PRIVATE ((__kernel_key_t) 0)
struct __kernel_legacy_ipc_perm {
  __kernel_key_t key;
  __kernel_uid_t uid;
  __kernel_gid_t gid;
  __kernel_uid_t cuid;
  __kernel_gid_t cgid;
  __kernel_mode_t mode;
  unsigned short seq;
};
#include <asm/ipcbuf.h>
#define IPC_CREAT 00001000
#define IPC_EXCL 00002000
#define IPC_NOWAIT 00004000
#define IPC_DIPC 00010000
#define IPC_OWN 00020000
#define IPC_RMID 0
#define IPC_SET 1
#define IPC_STAT 2
#define IPC_INFO 3
#define IPC_OLD 0
#define IPC_64 0x0100
struct ipc_kludge {
  struct msgbuf  * msgp;
  long msgtyp;
};
#define SEMOP 1
#define SEMGET 2
#define SEMCTL 3
#define SEMTIMEDOP 4
#define MSGSND 11
#define MSGRCV 12
#define MSGGET 13
#define MSGCTL 14
#define SHMAT 21
#define SHMDT 22
#define SHMGET 23
#define SHMCTL 24
#define DIPC 25
#define IPCCALL(version,op) ((version) << 16 | (op))
#endif
