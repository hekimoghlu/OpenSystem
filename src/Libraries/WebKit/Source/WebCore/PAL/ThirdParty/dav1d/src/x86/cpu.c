/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include <string.h>

#include "common/attributes.h"

#include "src/x86/cpu.h"

typedef struct {
    uint32_t eax, ebx, edx, ecx;
} CpuidRegisters;

void dav1d_cpu_cpuid(CpuidRegisters *regs, unsigned leaf, unsigned subleaf);
uint64_t dav1d_cpu_xgetbv(unsigned xcr);

#define X(reg, mask) (((reg) & (mask)) == (mask))

COLD unsigned dav1d_get_cpu_flags_x86(void) {
    union {
        CpuidRegisters r;
        struct {
            uint32_t max_leaf;
            char vendor[12];
        };
    } cpu;
    dav1d_cpu_cpuid(&cpu.r, 0, 0);
    unsigned flags = 0;

    if (cpu.max_leaf >= 1) {
        CpuidRegisters r;
        dav1d_cpu_cpuid(&r, 1, 0);
        const unsigned model  = ((r.eax >> 4) & 0x0f) + ((r.eax >> 12) & 0xf0);
        const unsigned family = ((r.eax >> 8) & 0x0f) + ((r.eax >> 20) & 0xff);

        if (X(r.edx, 0x06008000)) /* CMOV/SSE/SSE2 */ {
            flags |= DAV1D_X86_CPU_FLAG_SSE2;
            if (X(r.ecx, 0x00000201)) /* SSE3/SSSE3 */ {
                flags |= DAV1D_X86_CPU_FLAG_SSSE3;
                if (X(r.ecx, 0x00080000)) /* SSE4.1 */
                    flags |= DAV1D_X86_CPU_FLAG_SSE41;
            }
        }
#if ARCH_X86_64
        /* We only support >128-bit SIMD on x86-64. */
        if (X(r.ecx, 0x18000000)) /* OSXSAVE/AVX */ {
            const uint64_t xcr0 = dav1d_cpu_xgetbv(0);
            if (X(xcr0, 0x00000006)) /* XMM/YMM */ {
                if (cpu.max_leaf >= 7) {
                    dav1d_cpu_cpuid(&r, 7, 0);
                    if (X(r.ebx, 0x00000128)) /* BMI1/BMI2/AVX2 */ {
                        flags |= DAV1D_X86_CPU_FLAG_AVX2;
                        if (X(xcr0, 0x000000e0)) /* ZMM/OPMASK */ {
                            if (X(r.ebx, 0xd0230000) && X(r.ecx, 0x00005f42))
                                flags |= DAV1D_X86_CPU_FLAG_AVX512ICL;
                        }
                    }
                }
            }
        }
#endif
        if (!memcmp(cpu.vendor, "AuthenticAMD", sizeof(cpu.vendor))) {
            if ((flags & DAV1D_X86_CPU_FLAG_AVX2) && (family < 0x19 ||
                (family == 0x19 && (model < 0x10 || (model >= 0x20 && model < 0x60)))))
            {
                /* Excavator, Zen, Zen+, Zen 2, Zen 3, Zen 3+ */
                flags |= DAV1D_X86_CPU_FLAG_SLOW_GATHER;
            }
        }
    }

    return flags;
}
