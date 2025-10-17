/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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

#include <sys/times.h>

#include "header_checks.h"

static void sys_times_h() {
  TYPE(struct tms);
  STRUCT_MEMBER(struct tms, clock_t, tms_utime);
  STRUCT_MEMBER(struct tms, clock_t, tms_stime);
  STRUCT_MEMBER(struct tms, clock_t, tms_cutime);
  STRUCT_MEMBER(struct tms, clock_t, tms_cstime);

  TYPE(clock_t);

  FUNCTION(times, clock_t (*f)(struct tms*));
}
