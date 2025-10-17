/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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

#include <wtf/Platform.h>

#if CPU(ARM64)

#define RegisterNames ARM64Registers

#define FOR_EACH_REGISTER(macro)                \
    FOR_EACH_GP_REGISTER(macro)                 \
    FOR_EACH_FP_REGISTER(macro)

// We don't include LR in the set of callee-save registers even though it technically belongs
// there. This is because we use this set to describe the set of registers that need to be saved
// beyond what you would save by the platform-agnostic "preserve return address" and "restore
// return address" operations in CCallHelpers.

#if !OS(DARWIN)
#define FOR_EACH_GP_REGISTER(macro)                             \
    /* Parameter/result registers. */                           \
    macro(x0,  "x0"_s,  0, 0)                                     \
    macro(x1,  "x1"_s,  0, 0)                                     \
    macro(x2,  "x2"_s,  0, 0)                                     \
    macro(x3,  "x3"_s,  0, 0)                                     \
    macro(x4,  "x4"_s,  0, 0)                                     \
    macro(x5,  "x5"_s,  0, 0)                                     \
    macro(x6,  "x6"_s,  0, 0)                                     \
    macro(x7,  "x7"_s,  0, 0)                                     \
    /* Indirect result location register. */                    \
    macro(x8,  "x8"_s,  0, 0)                                     \
    /* Temporary registers. */                                  \
    macro(x9,  "x9"_s,  0, 0)                                     \
    macro(x10, "x10"_s, 0, 0)                                     \
    macro(x11, "x11"_s, 0, 0)                                     \
    macro(x12, "x12"_s, 0, 0)                                     \
    macro(x13, "x13"_s, 0, 0)                                     \
    macro(x14, "x14"_s, 0, 0)                                     \
    macro(x15, "x15"_s, 0, 0)                                     \
    /* Intra-procedure-call scratch registers (temporary). */   \
    macro(x16, "x16"_s, 0, 0)                                     \
    macro(x17, "x17"_s, 0, 0)                                     \
    /* Platform Register (temporary). */                        \
    macro(x18, "x18"_s, 0, 0)                                     \
    /* Callee-saved. */                                         \
    macro(x19, "x19"_s, 0, 1)                                     \
    macro(x20, "x20"_s, 0, 1)                                     \
    macro(x21, "x21"_s, 0, 1)                                     \
    macro(x22, "x22"_s, 0, 1)                                     \
    macro(x23, "x23"_s, 0, 1)                                     \
    macro(x24, "x24"_s, 0, 1)                                     \
    macro(x25, "x25"_s, 0, 1)                                     \
    macro(x26, "x26"_s, 0, 1)                                     \
    macro(x27, "x27"_s, 0, 1)                                     \
    macro(x28, "x28"_s, 0, 1)                                     \
    /* Special. */                                              \
    macro(fp,  "fp"_s,  0, 1)                                     \
    macro(lr,  "lr"_s,  1, 0)                                     \
    macro(sp,  "sp"_s,  0, 0)
#else
#define FOR_EACH_GP_REGISTER(macro)                             \
    /* Parameter/result registers. */                           \
    macro(x0,  "x0"_s,  0, 0)                                     \
    macro(x1,  "x1"_s,  0, 0)                                     \
    macro(x2,  "x2"_s,  0, 0)                                     \
    macro(x3,  "x3"_s,  0, 0)                                     \
    macro(x4,  "x4"_s,  0, 0)                                     \
    macro(x5,  "x5"_s,  0, 0)                                     \
    macro(x6,  "x6"_s,  0, 0)                                     \
    macro(x7,  "x7"_s,  0, 0)                                     \
    /* Indirect result location register. */                    \
    macro(x8,  "x8"_s,  0, 0)                                     \
    /* Temporary registers. */                                  \
    macro(x9,  "x9"_s,  0, 0)                                     \
    macro(x10, "x10"_s, 0, 0)                                     \
    macro(x11, "x11"_s, 0, 0)                                     \
    macro(x12, "x12"_s, 0, 0)                                     \
    macro(x13, "x13"_s, 0, 0)                                     \
    macro(x14, "x14"_s, 0, 0)                                     \
    macro(x15, "x15"_s, 0, 0)                                     \
    /* Intra-procedure-call scratch registers (temporary). */   \
    macro(x16, "x16"_s, 0, 0)                                     \
    macro(x17, "x17"_s, 0, 0)                                     \
    /* Platform Register (temporary). */                        \
    macro(x18, "x18"_s, 1, 0)                                     \
    /* Callee-saved. */                                         \
    macro(x19, "x19"_s, 0, 1)                                     \
    macro(x20, "x20"_s, 0, 1)                                     \
    macro(x21, "x21"_s, 0, 1)                                     \
    macro(x22, "x22"_s, 0, 1)                                     \
    macro(x23, "x23"_s, 0, 1)                                     \
    macro(x24, "x24"_s, 0, 1)                                     \
    macro(x25, "x25"_s, 0, 1)                                     \
    macro(x26, "x26"_s, 0, 1)                                     \
    macro(x27, "x27"_s, 0, 1)                                     \
    macro(x28, "x28"_s, 0, 1)                                     \
    /* Special. */                                              \
    macro(fp,  "fp"_s,  0, 1)                                     \
    macro(lr,  "lr"_s,  1, 0)                                     \
    macro(sp,  "sp"_s,  0, 0)
#endif

#define FOR_EACH_REGISTER_ALIAS(macro)          \
    macro(ip0, "ip0"_s, x16)                      \
    macro(ip1, "ip1"_s, x17)                      \
    macro(x29, "x29"_s, fp)                       \
    macro(x30, "x30"_s, lr)                       \
    macro(zr,  "zr"_s,  0x3f)

#define FOR_EACH_SP_REGISTER(macro)             \
    macro(pc,   "pc"_s)                           \
    macro(nzcv, "nzcv"_s)                         \
    macro(fpsr, "fpsr"_s)

#define FOR_EACH_FP_REGISTER(macro)             \
    /* Parameter/result registers. */           \
    macro(q0,  "q0"_s,  0, 0)                     \
    macro(q1,  "q1"_s,  0, 0)                     \
    macro(q2,  "q2"_s,  0, 0)                     \
    macro(q3,  "q3"_s,  0, 0)                     \
    macro(q4,  "q4"_s,  0, 0)                     \
    macro(q5,  "q5"_s,  0, 0)                     \
    macro(q6,  "q6"_s,  0, 0)                     \
    macro(q7,  "q7"_s,  0, 0)                     \
    /* Callee-saved (up to 64-bits only!). */   \
    macro(q8,  "q8"_s,  0, 1)                     \
    macro(q9,  "q9"_s,  0, 1)                     \
    macro(q10, "q10"_s, 0, 1)                     \
    macro(q11, "q11"_s, 0, 1)                     \
    macro(q12, "q12"_s, 0, 1)                     \
    macro(q13, "q13"_s, 0, 1)                     \
    macro(q14, "q14"_s, 0, 1)                     \
    macro(q15, "q15"_s, 0, 1)                     \
    /* Temporary registers. */                  \
    macro(q16, "q16"_s, 0, 0)                     \
    macro(q17, "q17"_s, 0, 0)                     \
    macro(q18, "q18"_s, 0, 0)                     \
    macro(q19, "q19"_s, 0, 0)                     \
    macro(q20, "q20"_s, 0, 0)                     \
    macro(q21, "q21"_s, 0, 0)                     \
    macro(q22, "q22"_s, 0, 0)                     \
    macro(q23, "q23"_s, 0, 0)                     \
    macro(q24, "q24"_s, 0, 0)                     \
    macro(q25, "q25"_s, 0, 0)                     \
    macro(q26, "q26"_s, 0, 0)                     \
    macro(q27, "q27"_s, 0, 0)                     \
    macro(q28, "q28"_s, 0, 0)                     \
    macro(q29, "q29"_s, 0, 0)                     \
    macro(q30, "q30"_s, 0, 0)                     \
    macro(q31, "q31"_s, 0, 0)

#endif // CPU(ARM64)
