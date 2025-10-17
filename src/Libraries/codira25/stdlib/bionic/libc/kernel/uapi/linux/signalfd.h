/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#ifndef _UAPI_LINUX_SIGNALFD_H
#define _UAPI_LINUX_SIGNALFD_H
#include <linux/types.h>
#include <linux/fcntl.h>
#define SFD_CLOEXEC O_CLOEXEC
#define SFD_NONBLOCK O_NONBLOCK
struct signalfd_siginfo {
  __u32 ssi_signo;
  __s32 ssi_errno;
  __s32 ssi_code;
  __u32 ssi_pid;
  __u32 ssi_uid;
  __s32 ssi_fd;
  __u32 ssi_tid;
  __u32 ssi_band;
  __u32 ssi_overrun;
  __u32 ssi_trapno;
  __s32 ssi_status;
  __s32 ssi_int;
  __u64 ssi_ptr;
  __u64 ssi_utime;
  __u64 ssi_stime;
  __u64 ssi_addr;
  __u16 ssi_addr_lsb;
  __u16 __pad2;
  __s32 ssi_syscall;
  __u64 ssi_call_addr;
  __u32 ssi_arch;
  __u8 __pad[28];
};
#endif
