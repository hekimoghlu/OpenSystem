/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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
#pragma once

/**
 * @file sys/shm.h
 * @brief System V shared memory. Not useful on Android because it's disallowed by SELinux.
 */

#include <sys/cdefs.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <unistd.h>

#include <linux/shm.h>

#define shmid_ds shmid64_ds
#define SHMLBA getpagesize()

__BEGIN_DECLS

typedef unsigned long shmatt_t;

/** Not useful on Android; disallowed by SELinux. */

#if __BIONIC_AVAILABILITY_GUARD(26)
void* _Nonnull shmat(int __shm_id, const void* _Nullable __addr, int __flags) __INTRODUCED_IN(26);
/** Not useful on Android; disallowed by SELinux. */
int shmctl(int __shm_id, int __op, struct shmid_ds* _Nullable __buf) __INTRODUCED_IN(26);
/** Not useful on Android; disallowed by SELinux. */
int shmdt(const void* _Nonnull __addr) __INTRODUCED_IN(26);
/** Not useful on Android; disallowed by SELinux. */
int shmget(key_t __key, size_t __size, int __flags) __INTRODUCED_IN(26);
#endif /* __BIONIC_AVAILABILITY_GUARD(26) */


__END_DECLS
