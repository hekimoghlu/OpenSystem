/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#include <sys/resource.h>

#include "header_checks.h"

static void sys_resource_h() {
  MACRO(PRIO_PROCESS);
  MACRO(PRIO_PGRP);
  MACRO(PRIO_USER);

  TYPE(rlim_t);

  MACRO(RLIM_INFINITY);
  MACRO(RLIM_SAVED_MAX);
  MACRO(RLIM_SAVED_CUR);

  MACRO(RUSAGE_SELF);
  MACRO(RUSAGE_CHILDREN);

  TYPE(struct rlimit);
  STRUCT_MEMBER(struct rlimit, rlim_t, rlim_cur);
  STRUCT_MEMBER(struct rlimit, rlim_t, rlim_max);

  TYPE(struct rusage);
  STRUCT_MEMBER(struct rusage, struct timeval, ru_utime);
  STRUCT_MEMBER(struct rusage, struct timeval, ru_stime);

  TYPE(struct timeval);

  MACRO(RLIMIT_CORE);
  MACRO(RLIMIT_CPU);
  MACRO(RLIMIT_DATA);
  MACRO(RLIMIT_FSIZE);
  MACRO(RLIMIT_NOFILE);
  MACRO(RLIMIT_STACK);
  MACRO(RLIMIT_AS);

  FUNCTION(getpriority, int (*f)(int, id_t));
  FUNCTION(getrlimit, int (*f)(int, struct rlimit*));
  FUNCTION(getrusage, int (*f)(int, struct rusage*));
  FUNCTION(setpriority, int (*f)(int, id_t, int));
  FUNCTION(setrlimit, int (*f)(int, const struct rlimit*));
}
