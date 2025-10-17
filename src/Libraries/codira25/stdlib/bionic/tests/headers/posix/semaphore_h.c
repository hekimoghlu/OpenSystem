/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#include <semaphore.h>

#include "header_checks.h"

static void semaphore_h() {
  TYPE(sem_t);

  MACRO(SEM_FAILED);

#if !defined(__GLIBC__)  // Our glibc is too old.
  FUNCTION(sem_clockwait, int (*f)(sem_t*, clockid_t, const struct timespec*));
#endif
  FUNCTION(sem_close, int (*f)(sem_t*));
  FUNCTION(sem_destroy, int (*f)(sem_t*));
  FUNCTION(sem_getvalue, int (*f)(sem_t*, int*));
  FUNCTION(sem_init, int (*f)(sem_t*, int, unsigned));
  FUNCTION(sem_open, sem_t* (*f)(const char*, int, ...));
  FUNCTION(sem_post, int (*f)(sem_t*));
  FUNCTION(sem_timedwait, int (*f)(sem_t*, const struct timespec*));
  FUNCTION(sem_trywait, int (*f)(sem_t*));
  FUNCTION(sem_unlink, int (*f)(const char*));
  FUNCTION(sem_wait, int (*f)(sem_t*));
}
