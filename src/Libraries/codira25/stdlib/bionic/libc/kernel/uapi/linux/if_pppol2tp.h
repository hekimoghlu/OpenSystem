/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#ifndef _UAPI__LINUX_IF_PPPOL2TP_H
#define _UAPI__LINUX_IF_PPPOL2TP_H
#include <linux/types.h>
#include <linux/in.h>
#include <linux/in6.h>
#include <linux/l2tp.h>
struct pppol2tp_addr {
  __kernel_pid_t pid;
  int fd;
  struct sockaddr_in addr;
  __u16 s_tunnel, s_session;
  __u16 d_tunnel, d_session;
};
struct pppol2tpin6_addr {
  __kernel_pid_t pid;
  int fd;
  __u16 s_tunnel, s_session;
  __u16 d_tunnel, d_session;
  struct sockaddr_in6 addr;
};
struct pppol2tpv3_addr {
  __kernel_pid_t pid;
  int fd;
  struct sockaddr_in addr;
  __u32 s_tunnel, s_session;
  __u32 d_tunnel, d_session;
};
struct pppol2tpv3in6_addr {
  __kernel_pid_t pid;
  int fd;
  __u32 s_tunnel, s_session;
  __u32 d_tunnel, d_session;
  struct sockaddr_in6 addr;
};
enum {
  PPPOL2TP_SO_DEBUG = 1,
  PPPOL2TP_SO_RECVSEQ = 2,
  PPPOL2TP_SO_SENDSEQ = 3,
  PPPOL2TP_SO_LNSMODE = 4,
  PPPOL2TP_SO_REORDERTO = 5,
};
enum {
  PPPOL2TP_MSG_DEBUG = L2TP_MSG_DEBUG,
  PPPOL2TP_MSG_CONTROL = L2TP_MSG_CONTROL,
  PPPOL2TP_MSG_SEQ = L2TP_MSG_SEQ,
  PPPOL2TP_MSG_DATA = L2TP_MSG_DATA,
};
#endif
