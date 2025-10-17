/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#ifndef __ASM_GENERIC_MSGBUF_H
#define __ASM_GENERIC_MSGBUF_H
#include <asm/bitsperlong.h>
#include <asm/ipcbuf.h>
struct msqid64_ds {
  struct ipc64_perm msg_perm;
#if __BITS_PER_LONG == 64
  long msg_stime;
  long msg_rtime;
  long msg_ctime;
#else
  unsigned long msg_stime;
  unsigned long msg_stime_high;
  unsigned long msg_rtime;
  unsigned long msg_rtime_high;
  unsigned long msg_ctime;
  unsigned long msg_ctime_high;
#endif
  unsigned long msg_cbytes;
  unsigned long msg_qnum;
  unsigned long msg_qbytes;
  __kernel_pid_t msg_lspid;
  __kernel_pid_t msg_lrpid;
  unsigned long __unused4;
  unsigned long __unused5;
};
#endif
