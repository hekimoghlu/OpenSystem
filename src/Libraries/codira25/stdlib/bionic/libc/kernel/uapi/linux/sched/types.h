/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#ifndef _UAPI_LINUX_SCHED_TYPES_H
#define _UAPI_LINUX_SCHED_TYPES_H
#include <linux/types.h>
#define SCHED_ATTR_SIZE_VER0 48
#define SCHED_ATTR_SIZE_VER1 56
struct sched_attr {
  __u32 size;
  __u32 sched_policy;
  __u64 sched_flags;
  __s32 sched_nice;
  __u32 sched_priority;
  __u64 sched_runtime;
  __u64 sched_deadline;
  __u64 sched_period;
  __u32 sched_util_min;
  __u32 sched_util_max;
};
#endif
