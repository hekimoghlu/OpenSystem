/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#ifndef __ASM_X64_MSGBUF_H
#define __ASM_X64_MSGBUF_H
#if !defined(__x86_64__) || !defined(__ILP32__)
#include <asm-generic/msgbuf.h>
#else
#include <asm/ipcbuf.h>
struct msqid64_ds {
  struct ipc64_perm msg_perm;
  __kernel_long_t msg_stime;
  __kernel_long_t msg_rtime;
  __kernel_long_t msg_ctime;
  __kernel_ulong_t msg_cbytes;
  __kernel_ulong_t msg_qnum;
  __kernel_ulong_t msg_qbytes;
  __kernel_pid_t msg_lspid;
  __kernel_pid_t msg_lrpid;
  __kernel_ulong_t __unused4;
  __kernel_ulong_t __unused5;
};
#endif
#endif
