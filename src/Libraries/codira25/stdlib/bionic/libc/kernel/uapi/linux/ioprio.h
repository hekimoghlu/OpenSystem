/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#ifndef _UAPI_LINUX_IOPRIO_H
#define _UAPI_LINUX_IOPRIO_H
#include <linux/stddef.h>
#include <linux/types.h>
#define IOPRIO_CLASS_SHIFT 13
#define IOPRIO_NR_CLASSES 8
#define IOPRIO_CLASS_MASK (IOPRIO_NR_CLASSES - 1)
#define IOPRIO_PRIO_MASK ((1UL << IOPRIO_CLASS_SHIFT) - 1)
#define IOPRIO_PRIO_CLASS(ioprio) (((ioprio) >> IOPRIO_CLASS_SHIFT) & IOPRIO_CLASS_MASK)
#define IOPRIO_PRIO_DATA(ioprio) ((ioprio) & IOPRIO_PRIO_MASK)
enum {
  IOPRIO_CLASS_NONE = 0,
  IOPRIO_CLASS_RT = 1,
  IOPRIO_CLASS_BE = 2,
  IOPRIO_CLASS_IDLE = 3,
  IOPRIO_CLASS_INVALID = 7,
};
#define IOPRIO_LEVEL_NR_BITS 3
#define IOPRIO_NR_LEVELS (1 << IOPRIO_LEVEL_NR_BITS)
#define IOPRIO_LEVEL_MASK (IOPRIO_NR_LEVELS - 1)
#define IOPRIO_PRIO_LEVEL(ioprio) ((ioprio) & IOPRIO_LEVEL_MASK)
#define IOPRIO_BE_NR IOPRIO_NR_LEVELS
enum {
  IOPRIO_WHO_PROCESS = 1,
  IOPRIO_WHO_PGRP,
  IOPRIO_WHO_USER,
};
#define IOPRIO_NORM 4
#define IOPRIO_BE_NORM IOPRIO_NORM
#define IOPRIO_HINT_SHIFT IOPRIO_LEVEL_NR_BITS
#define IOPRIO_HINT_NR_BITS 10
#define IOPRIO_NR_HINTS (1 << IOPRIO_HINT_NR_BITS)
#define IOPRIO_HINT_MASK (IOPRIO_NR_HINTS - 1)
#define IOPRIO_PRIO_HINT(ioprio) (((ioprio) >> IOPRIO_HINT_SHIFT) & IOPRIO_HINT_MASK)
enum {
  IOPRIO_HINT_NONE = 0,
  IOPRIO_HINT_DEV_DURATION_LIMIT_1 = 1,
  IOPRIO_HINT_DEV_DURATION_LIMIT_2 = 2,
  IOPRIO_HINT_DEV_DURATION_LIMIT_3 = 3,
  IOPRIO_HINT_DEV_DURATION_LIMIT_4 = 4,
  IOPRIO_HINT_DEV_DURATION_LIMIT_5 = 5,
  IOPRIO_HINT_DEV_DURATION_LIMIT_6 = 6,
  IOPRIO_HINT_DEV_DURATION_LIMIT_7 = 7,
};
#define IOPRIO_BAD_VALUE(val,max) ((val) < 0 || (val) >= (max))
static __always_inline __u16 ioprio_value(int prioclass, int priolevel, int priohint) {
  if(IOPRIO_BAD_VALUE(prioclass, IOPRIO_NR_CLASSES) || IOPRIO_BAD_VALUE(priolevel, IOPRIO_NR_LEVELS) || IOPRIO_BAD_VALUE(priohint, IOPRIO_NR_HINTS)) return IOPRIO_CLASS_INVALID << IOPRIO_CLASS_SHIFT;
  return(prioclass << IOPRIO_CLASS_SHIFT) | (priohint << IOPRIO_HINT_SHIFT) | priolevel;
}
#define IOPRIO_PRIO_VALUE(prioclass,priolevel) ioprio_value(prioclass, priolevel, IOPRIO_HINT_NONE)
#define IOPRIO_PRIO_VALUE_HINT(prioclass,priolevel,priohint) ioprio_value(prioclass, priolevel, priohint)
#endif
