/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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

#include "common/attributes.h"

#include "src/arm/cpu.h"

#if defined(__ARM_NEON) || defined(__APPLE__) || defined(_WIN32) || ARCH_AARCH64
// NEON is always available; runtime tests are not needed.
#elif defined(HAVE_GETAUXVAL) && ARCH_ARM
#include <sys/auxv.h>

#ifndef HWCAP_ARM_NEON
#define HWCAP_ARM_NEON (1 << 12)
#endif
#define NEON_HWCAP HWCAP_ARM_NEON

#elif defined(HAVE_ELF_AUX_INFO) && ARCH_ARM
#include <sys/auxv.h>

#define NEON_HWCAP HWCAP_NEON

#elif defined(__ANDROID__)
#include <stdio.h>
#include <string.h>

static unsigned parse_proc_cpuinfo(const char *flag) {
    FILE *file = fopen("/proc/cpuinfo", "r");
    if (!file)
        return 0;

    char line_buffer[120];
    const char *line;

    while ((line = fgets(line_buffer, sizeof(line_buffer), file))) {
        if (strstr(line, flag)) {
            fclose(file);
            return 1;
        }
        // if line is incomplete seek back to avoid splitting the search
        // string into two buffers
        if (!strchr(line, '\n') && strlen(line) > strlen(flag)) {
            // use fseek since the 64 bit fseeko is only available since
            // Android API level 24 and meson defines _FILE_OFFSET_BITS
            // by default 64
            if (fseek(file, -strlen(flag), SEEK_CUR))
                break;
        }
    }

    fclose(file);

    return 0;
}
#endif

COLD unsigned dav1d_get_cpu_flags_arm(void) {
    unsigned flags = 0;
#if defined(__ARM_NEON) || defined(__APPLE__) || defined(_WIN32) || ARCH_AARCH64
    flags |= DAV1D_ARM_CPU_FLAG_NEON;
#elif defined(HAVE_GETAUXVAL) && ARCH_ARM
    unsigned long hw_cap = getauxval(AT_HWCAP);
    flags |= (hw_cap & NEON_HWCAP) ? DAV1D_ARM_CPU_FLAG_NEON : 0;
#elif defined(HAVE_ELF_AUX_INFO) && ARCH_ARM
    unsigned long hw_cap = 0;
    elf_aux_info(AT_HWCAP, &hw_cap, sizeof(hw_cap));
    flags |= (hw_cap & NEON_HWCAP) ? DAV1D_ARM_CPU_FLAG_NEON : 0;
#elif defined(__ANDROID__)
    flags |= parse_proc_cpuinfo("neon") ? DAV1D_ARM_CPU_FLAG_NEON : 0;
#endif

    return flags;
}
