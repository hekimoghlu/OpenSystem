/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#ifndef DAV1D_SRC_X86_CPU_H
#define DAV1D_SRC_X86_CPU_H

enum CpuFlags {
    DAV1D_X86_CPU_FLAG_SSE2        = 1 << 0,
    DAV1D_X86_CPU_FLAG_SSSE3       = 1 << 1,
    DAV1D_X86_CPU_FLAG_SSE41       = 1 << 2,
    DAV1D_X86_CPU_FLAG_AVX2        = 1 << 3,
    DAV1D_X86_CPU_FLAG_AVX512ICL   = 1 << 4, /* F/CD/BW/DQ/VL/VNNI/IFMA/VBMI/VBMI2/
                                              * VPOPCNTDQ/BITALG/GFNI/VAES/VPCLMULQDQ */
    DAV1D_X86_CPU_FLAG_SLOW_GATHER = 1 << 5, /* Flag CPUs where gather instructions are slow enough
                                              * to cause performance regressions. */
};

unsigned dav1d_get_cpu_flags_x86(void);

#endif /* DAV1D_SRC_X86_CPU_H */
