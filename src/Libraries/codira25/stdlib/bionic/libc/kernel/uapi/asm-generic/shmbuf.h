/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#ifndef __ASM_GENERIC_SHMBUF_H
#define __ASM_GENERIC_SHMBUF_H
#include <asm/bitsperlong.h>
#include <asm/ipcbuf.h>
#include <asm/posix_types.h>
struct shmid64_ds {
  struct ipc64_perm shm_perm;
  __kernel_size_t shm_segsz;
#if __BITS_PER_LONG == 64
  long shm_atime;
  long shm_dtime;
  long shm_ctime;
#else
  unsigned long shm_atime;
  unsigned long shm_atime_high;
  unsigned long shm_dtime;
  unsigned long shm_dtime_high;
  unsigned long shm_ctime;
  unsigned long shm_ctime_high;
#endif
  __kernel_pid_t shm_cpid;
  __kernel_pid_t shm_lpid;
  unsigned long shm_nattch;
  unsigned long __unused4;
  unsigned long __unused5;
};
struct shminfo64 {
  unsigned long shmmax;
  unsigned long shmmin;
  unsigned long shmmni;
  unsigned long shmseg;
  unsigned long shmall;
  unsigned long __unused1;
  unsigned long __unused2;
  unsigned long __unused3;
  unsigned long __unused4;
};
#endif
