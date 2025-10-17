/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#if defined(__BIONIC__)

#include <sys/sem.h>

#include "header_checks.h"

static void sys_sem_h() {
  MACRO(SEM_UNDO);

  MACRO(GETNCNT);
  MACRO(GETPID);
  MACRO(GETVAL);
  MACRO(GETALL);
  MACRO(GETZCNT);
  MACRO(SETVAL);
  MACRO(SETALL);

  TYPE(struct semid_ds);
  STRUCT_MEMBER(struct semid_ds, struct ipc_perm, sem_perm);
#if defined(__linux__) // POSIX says short, Linux says long.
  STRUCT_MEMBER(struct semid_ds, unsigned long, sem_nsems);
#else
  STRUCT_MEMBER(struct semid_ds, unsigned short, sem_nsems);
#endif
#if defined(__LP64__)
  STRUCT_MEMBER(struct semid_ds, time_t, sem_otime);
  STRUCT_MEMBER(struct semid_ds, time_t, sem_ctime);
#else
  // Starting at kernel v4.19, 32 bit changed these to unsigned values.
  STRUCT_MEMBER(struct semid_ds, unsigned long, sem_otime);
  STRUCT_MEMBER(struct semid_ds, unsigned long, sem_ctime);
#endif

  TYPE(pid_t);
  TYPE(size_t);
  TYPE(time_t);

  TYPE(struct sembuf);
  STRUCT_MEMBER(struct sembuf, unsigned short, sem_num);
  STRUCT_MEMBER(struct sembuf, short, sem_op);
  STRUCT_MEMBER(struct sembuf, short, sem_flg);

  FUNCTION(semctl, int (*f)(int, int, int, ...));
  FUNCTION(semget, int (*f)(key_t, int, int));
  FUNCTION(semop, int (*f)(int, struct sembuf*, size_t));
}

#endif
