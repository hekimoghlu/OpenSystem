/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#ifndef _UAPI_LINUX_RESOURCE_H
#define _UAPI_LINUX_RESOURCE_H
#include <linux/time_types.h>
#include <linux/types.h>
#define RUSAGE_SELF 0
#define RUSAGE_CHILDREN (- 1)
#define RUSAGE_BOTH (- 2)
#define RUSAGE_THREAD 1
struct rusage {
  struct timeval ru_utime;
  struct timeval ru_stime;
  __kernel_long_t ru_maxrss;
  __kernel_long_t ru_ixrss;
  __kernel_long_t ru_idrss;
  __kernel_long_t ru_isrss;
  __kernel_long_t ru_minflt;
  __kernel_long_t ru_majflt;
  __kernel_long_t ru_nswap;
  __kernel_long_t ru_inblock;
  __kernel_long_t ru_oublock;
  __kernel_long_t ru_msgsnd;
  __kernel_long_t ru_msgrcv;
  __kernel_long_t ru_nsignals;
  __kernel_long_t ru_nvcsw;
  __kernel_long_t ru_nivcsw;
};
struct rlimit {
  __kernel_ulong_t rlim_cur;
  __kernel_ulong_t rlim_max;
};
#define RLIM64_INFINITY (~0ULL)
struct rlimit64 {
  __u64 rlim_cur;
  __u64 rlim_max;
};
#define PRIO_MIN (- 20)
#define PRIO_MAX 20
#define PRIO_PROCESS 0
#define PRIO_PGRP 1
#define PRIO_USER 2
#define _STK_LIM (8 * 1024 * 1024)
#define MLOCK_LIMIT (8 * 1024 * 1024)
#include <asm/resource.h>
#endif
