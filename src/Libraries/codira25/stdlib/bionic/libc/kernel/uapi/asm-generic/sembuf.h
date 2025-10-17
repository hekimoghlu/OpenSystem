/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
#ifndef __ASM_GENERIC_SEMBUF_H
#define __ASM_GENERIC_SEMBUF_H
#include <asm/bitsperlong.h>
#include <asm/ipcbuf.h>
struct semid64_ds {
  struct ipc64_perm sem_perm;
#if __BITS_PER_LONG == 64
  long sem_otime;
  long sem_ctime;
#else
  unsigned long sem_otime;
  unsigned long sem_otime_high;
  unsigned long sem_ctime;
  unsigned long sem_ctime_high;
#endif
  unsigned long sem_nsems;
  unsigned long __unused3;
  unsigned long __unused4;
};
#endif
