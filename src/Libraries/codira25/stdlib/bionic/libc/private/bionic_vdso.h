/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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

#if defined(__aarch64__)
#define VDSO_CLOCK_GETTIME_SYMBOL "__kernel_clock_gettime"
#define VDSO_CLOCK_GETRES_SYMBOL "__kernel_clock_getres"
#define VDSO_GETTIMEOFDAY_SYMBOL "__kernel_gettimeofday"
#else
#define VDSO_CLOCK_GETTIME_SYMBOL "__vdso_clock_gettime"
#define VDSO_CLOCK_GETRES_SYMBOL "__vdso_clock_getres"
#define VDSO_GETTIMEOFDAY_SYMBOL "__vdso_gettimeofday"
#endif
#if defined(__riscv)
#define VDSO_RISCV_HWPROBE_SYMBOL "__vdso_riscv_hwprobe"
#endif
#if defined(__i386__) || defined(__x86_64__)
#define VDSO_TIME_SYMBOL "__vdso_time"
#endif

struct vdso_entry {
  const char* name;
  void* fn;
};

enum {
  VDSO_CLOCK_GETTIME = 0,
  VDSO_CLOCK_GETRES,
  VDSO_GETTIMEOFDAY,
#if defined(VDSO_TIME_SYMBOL)
  VDSO_TIME,
#endif
#if defined(VDSO_RISCV_HWPROBE_SYMBOL)
  VDSO_RISCV_HWPROBE,
#endif
  VDSO_END
};
