/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "vpx_config.h"
#include "vp8_rtcd.h"
#if VPX_ARCH_ARM
#include "vpx_ports/arm.h"
#elif VPX_ARCH_X86 || VPX_ARCH_X86_64
#include "vpx_ports/x86.h"
#elif VPX_ARCH_PPC
#include "vpx_ports/ppc.h"
#elif VPX_ARCH_MIPS
#include "vpx_ports/mips.h"
#elif VPX_ARCH_LOONGARCH
#include "vpx_ports/loongarch.h"
#endif
#include "vp8/common/onyxc_int.h"
#include "vp8/common/systemdependent.h"

#if CONFIG_MULTITHREAD
#if HAVE_UNISTD_H
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
typedef void(WINAPI *PGNSI)(LPSYSTEM_INFO);
#endif
#endif

#if CONFIG_MULTITHREAD
static int get_cpu_count(void) {
  int core_count = 16;

#if HAVE_UNISTD_H
#if defined(_SC_NPROCESSORS_ONLN)
  core_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_SC_NPROC_ONLN)
  core_count = (int)sysconf(_SC_NPROC_ONLN);
#endif
#elif defined(_WIN32)
  {
#if _WIN32_WINNT < 0x0501
#error _WIN32_WINNT must target Windows XP or newer.
#endif
    SYSTEM_INFO sysinfo;
    GetNativeSystemInfo(&sysinfo);
    core_count = (int)sysinfo.dwNumberOfProcessors;
  }
#else
/* other platforms */
#endif

  return core_count > 0 ? core_count : 1;
}
#endif

void vp8_machine_specific_config(VP8_COMMON *ctx) {
#if CONFIG_MULTITHREAD
  ctx->processor_core_count = get_cpu_count();
#else
  (void)ctx;
#endif /* CONFIG_MULTITHREAD */
}
