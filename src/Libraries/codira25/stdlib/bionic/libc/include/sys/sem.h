/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#ifndef _SYS_SEM_H_
#define _SYS_SEM_H_

#include <sys/cdefs.h>
#include <sys/ipc.h>
#include <sys/types.h>

#if defined(__USE_GNU)
#include <bits/timespec.h>
#endif

#include <linux/sem.h>

__BEGIN_DECLS

#define semid_ds semid64_ds

union semun {
  int val;
  struct semid_ds* _Nullable buf;
  unsigned short* _Nullable array;
  struct seminfo* _Nullable __buf;
  void* _Nullable __pad;
};


#if __BIONIC_AVAILABILITY_GUARD(26)
int semctl(int __sem_id, int __sem_num, int __op, ...) __INTRODUCED_IN(26);
int semget(key_t __key, int __sem_count, int __flags) __INTRODUCED_IN(26);
int semop(int __sem_id, struct sembuf* _Nonnull __ops, size_t __op_count) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


#if defined(__USE_GNU)

#if __BIONIC_AVAILABILITY_GUARD(26)
int semtimedop(int __sem_id, struct sembuf* _Nonnull __ops, size_t __op_count, const struct timespec* _Nullable __timeout) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */

#endif

__END_DECLS

#endif
