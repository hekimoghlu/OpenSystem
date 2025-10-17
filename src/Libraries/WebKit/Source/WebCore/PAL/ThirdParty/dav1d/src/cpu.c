/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
#include "config.h"

#include <stdint.h>

#include "src/cpu.h"
#include "src/log.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#ifdef HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif
#if defined(__FreeBSD__)
#define cpu_set_t cpuset_t
#endif

static unsigned flags = 0;
static unsigned flags_mask = -1;

COLD void dav1d_init_cpu(void) {
#if HAVE_ASM && !__has_feature(memory_sanitizer)
// memory sanitizer is inherently incompatible with asm
#if ARCH_AARCH64 || ARCH_ARM
    flags = dav1d_get_cpu_flags_arm();
#elif ARCH_PPC64LE
    flags = dav1d_get_cpu_flags_ppc();
#elif ARCH_X86
    flags = dav1d_get_cpu_flags_x86();
#endif
#endif
}

COLD unsigned dav1d_get_cpu_flags(void) {
    return flags & flags_mask;
}

COLD void dav1d_set_cpu_flags_mask(const unsigned mask) {
    flags_mask = mask;
}

COLD int dav1d_num_logical_processors(Dav1dContext *const c) {
#ifdef _WIN32
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
    GROUP_AFFINITY affinity;
    if (GetThreadGroupAffinity(GetCurrentThread(), &affinity)) {
        int num_processors = 1;
        while (affinity.Mask &= affinity.Mask - 1)
            num_processors++;
        return num_processors;
    }
#else
    SYSTEM_INFO system_info;
    GetNativeSystemInfo(&system_info);
    return system_info.dwNumberOfProcessors;
#endif
#elif defined(HAVE_PTHREAD_GETAFFINITY_NP) && defined(CPU_COUNT)
    cpu_set_t affinity;
    if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity))
        return CPU_COUNT(&affinity);
#elif defined(__APPLE__)
    int num_processors;
    size_t length = sizeof(num_processors);
    if (!sysctlbyname("hw.logicalcpu", &num_processors, &length, NULL, 0))
        return num_processors;
#elif defined(_SC_NPROCESSORS_ONLN)
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    dav1d_log(c, "Unable to detect thread count, defaulting to single-threaded mode\n");
    return 1;
}
