/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#ifndef _LINUX_SYSINFO_H
#define _LINUX_SYSINFO_H
#include <linux/types.h>
#define SI_LOAD_SHIFT 16
struct sysinfo {
  __kernel_long_t uptime;
  __kernel_ulong_t loads[3];
  __kernel_ulong_t totalram;
  __kernel_ulong_t freeram;
  __kernel_ulong_t sharedram;
  __kernel_ulong_t bufferram;
  __kernel_ulong_t totalswap;
  __kernel_ulong_t freeswap;
  __u16 procs;
  __u16 pad;
  __kernel_ulong_t totalhigh;
  __kernel_ulong_t freehigh;
  __u32 mem_unit;
  char _f[20 - 2 * sizeof(__kernel_ulong_t) - sizeof(__u32)];
};
#endif
