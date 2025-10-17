/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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

#if CPU(RISCV64)

// More on the RISC-V calling convention and registers:
// https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf
// https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf (Chapter 20)

#define RegisterNames RISCV64Registers

#define FOR_EACH_GP_REGISTER(macro)    \
    macro(x0, "x0"_s, 1, 0)              \
    macro(x1, "x1"_s, 1, 0)              \
    macro(x2, "x2"_s, 1, 1)              \
    macro(x3, "x3"_s, 1, 0)              \
    macro(x4, "x4"_s, 1, 0)              \
    macro(x5, "x5"_s, 0, 0)              \
    macro(x6, "x6"_s, 0, 0)              \
    macro(x7, "x7"_s, 0, 0)              \
    macro(x8, "x8"_s, 1, 1)              \
    macro(x9, "x9"_s, 0, 1)              \
    macro(x10, "x10"_s, 0, 0)            \
    macro(x11, "x11"_s, 0, 0)            \
    macro(x12, "x12"_s, 0, 0)            \
    macro(x13, "x13"_s, 0, 0)            \
    macro(x14, "x14"_s, 0, 0)            \
    macro(x15, "x15"_s, 0, 0)            \
    macro(x16, "x16"_s, 0, 0)            \
    macro(x17, "x17"_s, 0, 0)            \
    macro(x18, "x18"_s, 0, 1)            \
    macro(x19, "x19"_s, 0, 1)            \
    macro(x20, "x20"_s, 0, 1)            \
    macro(x21, "x21"_s, 0, 1)            \
    macro(x22, "x22"_s, 0, 1)            \
    macro(x23, "x23"_s, 0, 1)            \
    macro(x24, "x24"_s, 0, 1)            \
    macro(x25, "x25"_s, 0, 1)            \
    macro(x26, "x26"_s, 0, 1)            \
    macro(x27, "x27"_s, 0, 1)            \
    macro(x28, "x28"_s, 0, 0)            \
    macro(x29, "x29"_s, 0, 0)            \
/* MacroAssembler scratch registers */ \
    macro(x30, "x30"_s, 0, 0)            \
    macro(x31, "x31"_s, 0, 0)

#define FOR_EACH_REGISTER_ALIAS(macro) \
    macro(zero, "zero"_s, x0)            \
    macro(ra, "ra"_s, x1)                \
    macro(sp, "sp"_s, x2)                \
    macro(gp, "gp"_s, x3)                \
    macro(tp, "tp"_s, x4)                \
    macro(fp, "fp"_s, x8)

#define FOR_EACH_SP_REGISTER(macro) \
    macro(pc, "pc"_s)

#define FOR_EACH_FP_REGISTER(macro) \
    macro(f0, "f0"_s, 0, 0)           \
    macro(f1, "f1"_s, 0, 0)           \
    macro(f2, "f2"_s, 0, 0)           \
    macro(f3, "f3"_s, 0, 0)           \
    macro(f4, "f4"_s, 0, 0)           \
    macro(f5, "f5"_s, 0, 0)           \
    macro(f6, "f6"_s, 0, 0)           \
    macro(f7, "f7"_s, 0, 0)           \
    macro(f8, "f8"_s, 0, 1)           \
    macro(f9, "f9"_s, 0, 1)           \
    macro(f10, "f10"_s, 0, 0)         \
    macro(f11, "f11"_s, 0, 0)         \
    macro(f12, "f12"_s, 0, 0)         \
    macro(f13, "f13"_s, 0, 0)         \
    macro(f14, "f14"_s, 0, 0)         \
    macro(f15, "f15"_s, 0, 0)         \
    macro(f16, "f16"_s, 0, 0)         \
    macro(f17, "f17"_s, 0, 0)         \
    macro(f18, "f18"_s, 0, 1)         \
    macro(f19, "f19"_s, 0, 1)         \
    macro(f20, "f20"_s, 0, 1)         \
    macro(f21, "f21"_s, 0, 1)         \
    macro(f22, "f22"_s, 0, 1)         \
    macro(f23, "f23"_s, 0, 1)         \
    macro(f24, "f24"_s, 0, 1)         \
    macro(f25, "f25"_s, 0, 1)         \
    macro(f26, "f26"_s, 0, 1)         \
    macro(f27, "f27"_s, 0, 1)         \
    macro(f28, "f28"_s, 0, 0)         \
    macro(f29, "f29"_s, 0, 0)         \
    macro(f30, "f30"_s, 0, 0)         \
    macro(f31, "f31"_s, 0, 0)

#endif // CPU(RISCV64)
