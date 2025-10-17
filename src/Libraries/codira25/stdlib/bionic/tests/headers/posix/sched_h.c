/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

#if !defined(DO_NOT_INCLUDE_SCHED_H)
#include <sched.h>
#endif

#include "header_checks.h"

static void sched_h() {
  TYPE(pid_t);
  TYPE(time_t);
  TYPE(struct timespec);

  TYPE(struct sched_param);
  STRUCT_MEMBER(struct sched_param, int, sched_priority);
#if !defined(__linux__)
  STRUCT_MEMBER(struct sched_param, int, sched_ss_low_priority);
  STRUCT_MEMBER(struct sched_param, struct timespec, sched_ss_repl_period);
  STRUCT_MEMBER(struct sched_param, struct timespec, sched_ss_init_budget);
  STRUCT_MEMBER(struct sched_param, int, sched_ss_max_repl);
#endif

  MACRO(SCHED_FIFO);
  MACRO(SCHED_RR);
#if !defined(__linux__)
  MACRO(SCHED_SPORADIC);
#endif
  MACRO(SCHED_OTHER);

  FUNCTION(sched_get_priority_max, int (*f)(int));
  FUNCTION(sched_get_priority_min, int (*f)(int));
  FUNCTION(sched_getparam, int (*f)(pid_t, struct sched_param*));
  FUNCTION(sched_getscheduler, int (*f)(pid_t));
  FUNCTION(sched_rr_get_interval, int (*f)(pid_t, struct timespec*));
  FUNCTION(sched_setparam, int (*f)(pid_t, const struct sched_param*));
  FUNCTION(sched_setscheduler, int (*f)(pid_t, int, const struct sched_param*));
  FUNCTION(sched_yield, int (*f)(void));
}
