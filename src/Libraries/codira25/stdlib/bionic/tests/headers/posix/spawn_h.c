/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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

#include <spawn.h>

#include "header_checks.h"

static void spawn_h() {
  TYPE(posix_spawnattr_t);
  TYPE(posix_spawn_file_actions_t);

  TYPE(mode_t);
  TYPE(pid_t);

  TYPE(sigset_t);

  TYPE(struct sched_param*);

  MACRO(POSIX_SPAWN_RESETIDS);
  MACRO(POSIX_SPAWN_SETPGROUP);
  MACRO(POSIX_SPAWN_SETSCHEDPARAM);
  MACRO(POSIX_SPAWN_SETSCHEDULER);
  MACRO(POSIX_SPAWN_SETSIGDEF);
  MACRO(POSIX_SPAWN_SETSIGMASK);

  FUNCTION(posix_spawn, int (*f)(pid_t*, const char*, const posix_spawn_file_actions_t*, const posix_spawnattr_t*, char* const[], char* const[]));
  FUNCTION(posix_spawn_file_actions_addclose, int (*f)(posix_spawn_file_actions_t*, int));
  FUNCTION(posix_spawn_file_actions_adddup2, int (*f)(posix_spawn_file_actions_t*, int, int));
  FUNCTION(posix_spawn_file_actions_addopen, int (*f)(posix_spawn_file_actions_t*, int, const char*, int, mode_t));
  FUNCTION(posix_spawn_file_actions_destroy, int (*f)(posix_spawn_file_actions_t*));
  FUNCTION(posix_spawn_file_actions_init, int (*f)(posix_spawn_file_actions_t*));
  FUNCTION(posix_spawnattr_destroy, int (*f)(posix_spawnattr_t*));
  FUNCTION(posix_spawnattr_getflags, int (*f)(const posix_spawnattr_t*, short*));
  FUNCTION(posix_spawnattr_getpgroup, int (*f)(const posix_spawnattr_t*, pid_t*));
  FUNCTION(posix_spawnattr_getschedparam, int (*f)(const posix_spawnattr_t*, struct sched_param*));
  FUNCTION(posix_spawnattr_getschedpolicy, int (*f)(const posix_spawnattr_t*, int*));
  FUNCTION(posix_spawnattr_getsigdefault, int (*f)(const posix_spawnattr_t*, sigset_t*));
  FUNCTION(posix_spawnattr_getsigmask, int (*f)(const posix_spawnattr_t*, sigset_t*));
  FUNCTION(posix_spawnattr_init, int (*f)(posix_spawnattr_t*));
  FUNCTION(posix_spawnattr_setflags, int (*f)(posix_spawnattr_t*, short));
  FUNCTION(posix_spawnattr_setpgroup, int (*f)(posix_spawnattr_t*, pid_t));
  FUNCTION(posix_spawnattr_setschedparam, int (*f)(posix_spawnattr_t*, const struct sched_param*));
  FUNCTION(posix_spawnattr_setsigdefault, int (*f)(posix_spawnattr_t*, const sigset_t*));
  FUNCTION(posix_spawnattr_setsigmask, int (*f)(posix_spawnattr_t*, const sigset_t*));
  FUNCTION(posix_spawnp, int (*f)(pid_t*, const char*, const posix_spawn_file_actions_t*, const posix_spawnattr_t*, char* const[], char* const[]));
}
