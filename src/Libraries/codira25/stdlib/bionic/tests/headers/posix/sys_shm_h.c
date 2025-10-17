/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include <sys/shm.h>

#include "header_checks.h"

static void sys_shm_h() {
  MACRO(SHM_RDONLY);
  MACRO(SHM_RND);
  MACRO(SHMLBA);

  TYPE(shmatt_t);

  TYPE(struct shmid_ds);
  STRUCT_MEMBER(struct shmid_ds, struct ipc_perm, shm_perm);
  STRUCT_MEMBER(struct shmid_ds, size_t, shm_segsz);
  STRUCT_MEMBER(struct shmid_ds, pid_t, shm_lpid);
  STRUCT_MEMBER(struct shmid_ds, pid_t, shm_cpid);
  STRUCT_MEMBER(struct shmid_ds, shmatt_t, shm_nattch);
#if defined(__LP64__)
  STRUCT_MEMBER(struct shmid_ds, time_t, shm_atime);
  STRUCT_MEMBER(struct shmid_ds, time_t, shm_dtime);
  STRUCT_MEMBER(struct shmid_ds, time_t, shm_ctime);
#else
  // Starting at kernel v4.19, 32 bit changed these to unsigned values.
  STRUCT_MEMBER(struct shmid_ds, unsigned long, shm_atime);
  STRUCT_MEMBER(struct shmid_ds, unsigned long, shm_dtime);
  STRUCT_MEMBER(struct shmid_ds, unsigned long, shm_ctime);
#endif

  TYPE(pid_t);
  TYPE(size_t);
  TYPE(time_t);

  FUNCTION(shmat, void* (*f)(int, const void*, int));
  FUNCTION(shmctl, int (*f)(int, int, struct shmid_ds*));
  FUNCTION(shmdt, int (*f)(const void*));
  FUNCTION(shmget, int (*f)(key_t, size_t, int));
}
#endif
