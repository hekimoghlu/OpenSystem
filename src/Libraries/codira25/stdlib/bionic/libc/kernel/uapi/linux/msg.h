/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
#ifndef _UAPI_LINUX_MSG_H
#define _UAPI_LINUX_MSG_H
#include <linux/ipc.h>
#define MSG_STAT 11
#define MSG_INFO 12
#define MSG_STAT_ANY 13
#define MSG_NOERROR 010000
#define MSG_EXCEPT 020000
#define MSG_COPY 040000
struct __kernel_legacy_msqid_ds {
  struct __kernel_legacy_ipc_perm msg_perm;
  struct msg * msg_first;
  struct msg * msg_last;
  __kernel_old_time_t msg_stime;
  __kernel_old_time_t msg_rtime;
  __kernel_old_time_t msg_ctime;
  unsigned long msg_lcbytes;
  unsigned long msg_lqbytes;
  unsigned short msg_cbytes;
  unsigned short msg_qnum;
  unsigned short msg_qbytes;
  __kernel_ipc_pid_t msg_lspid;
  __kernel_ipc_pid_t msg_lrpid;
};
#include <asm/msgbuf.h>
struct msgbuf {
  __kernel_long_t mtype;
  char mtext[1];
};
struct msginfo {
  int msgpool;
  int msgmap;
  int msgmax;
  int msgmnb;
  int msgmni;
  int msgssz;
  int msgtql;
  unsigned short msgseg;
};
#define MSGMNI 32000
#define MSGMAX 8192
#define MSGMNB 16384
#define MSGPOOL (MSGMNI * MSGMNB / 1024)
#define MSGTQL MSGMNB
#define MSGMAP MSGMNB
#define MSGSSZ 16
#define __MSGSEG ((MSGPOOL * 1024) / MSGSSZ)
#define MSGSEG (__MSGSEG <= 0xffff ? __MSGSEG : 0xffff)
#endif
