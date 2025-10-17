/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef __ASM_X86_SHMBUF_H
#define __ASM_X86_SHMBUF_H
#if !defined(__x86_64__) || !defined(__ILP32__)
#include <asm-generic/shmbuf.h>
#else
#include <asm/ipcbuf.h>
#include <asm/posix_types.h>
struct shmid64_ds {
  struct ipc64_perm shm_perm;
  __kernel_size_t shm_segsz;
  __kernel_long_t shm_atime;
  __kernel_long_t shm_dtime;
  __kernel_long_t shm_ctime;
  __kernel_pid_t shm_cpid;
  __kernel_pid_t shm_lpid;
  __kernel_ulong_t shm_nattch;
  __kernel_ulong_t __unused4;
  __kernel_ulong_t __unused5;
};
struct shminfo64 {
  __kernel_ulong_t shmmax;
  __kernel_ulong_t shmmin;
  __kernel_ulong_t shmmni;
  __kernel_ulong_t shmseg;
  __kernel_ulong_t shmall;
  __kernel_ulong_t __unused1;
  __kernel_ulong_t __unused2;
  __kernel_ulong_t __unused3;
  __kernel_ulong_t __unused4;
};
#endif
#endif
