/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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

#include <sys/cdefs.h>
#include <sys/types.h>

typedef struct {
  uint32_t flags;
  void* stack_base;
  size_t stack_size;
  size_t guard_size;
  int32_t sched_policy;
  int32_t sched_priority;
#ifdef __LP64__
  char __reserved[16];
#endif
} pthread_attr_t;

typedef struct {
#if defined(__LP64__)
  int64_t __private[4];
#else
  int32_t __private[8];
#endif
} pthread_barrier_t;

typedef int pthread_barrierattr_t;

typedef struct {
#if defined(__LP64__)
  int32_t __private[12];
#else
  int32_t __private[1];
#endif
} pthread_cond_t;

typedef long pthread_condattr_t;

typedef int pthread_key_t;

typedef struct {
#if defined(__LP64__)
  int32_t __private[10];
#else
  int32_t __private[1];
#endif
} pthread_mutex_t;

typedef long pthread_mutexattr_t;

typedef int pthread_once_t;

typedef struct {
#if defined(__LP64__)
  int32_t __private[14];
#else
  int32_t __private[10];
#endif
} pthread_rwlock_t;

typedef long pthread_rwlockattr_t;

typedef struct {
#if defined(__LP64__)
  int64_t __private;
#else
  int32_t __private[2];
#endif
} pthread_spinlock_t;

typedef long pthread_t;
