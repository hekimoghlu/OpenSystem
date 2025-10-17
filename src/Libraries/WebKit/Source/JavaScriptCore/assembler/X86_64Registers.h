/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#include <wtf/Compiler.h>
#include <wtf/Platform.h>

#if CPU(X86_64)

#define RegisterNames X86Registers

#define FOR_EACH_GP_REGISTER(macro)             \
    macro(eax, "rax"_s, 0, 0)                     \
    macro(ecx, "rcx"_s, 0, 0)                     \
    macro(edx, "rdx"_s, 0, 0)                     \
    macro(ebx, "rbx"_s, 0, 1)                     \
    macro(esp, "rsp"_s, 0, 0)                     \
    macro(ebp, "rbp"_s, 0, 1)                     \
    macro(esi, "rsi"_s, 0, 0)                     \
    macro(edi, "rdi"_s, 0, 0)                     \
    macro(r8,  "r8"_s,  0, 0)                     \
    macro(r9,  "r9"_s,  0, 0)                     \
    macro(r10, "r10"_s, 0, 0)                     \
    macro(r11, "r11"_s, 0, 0)                     \
    macro(r12, "r12"_s, 0, 1)                     \
    macro(r13, "r13"_s, 0, 1)                     \
    macro(r14, "r14"_s, 0, 1)                     \
    macro(r15, "r15"_s, 0, 1)

#define FOR_EACH_FP_REGISTER(macro)             \
    macro(xmm0,  "xmm0"_s,  0, 0)                  \
    macro(xmm1,  "xmm1"_s,  0, 0)                  \
    macro(xmm2,  "xmm2"_s,  0, 0)                  \
    macro(xmm3,  "xmm3"_s,  0, 0)                  \
    macro(xmm4,  "xmm4"_s,  0, 0)                  \
    macro(xmm5,  "xmm5"_s,  0, 0)                  \
    macro(xmm6,  "xmm6"_s,  0, 0)                  \
    macro(xmm7,  "xmm7"_s,  0, 0)                  \
    macro(xmm8,  "xmm8"_s,  0, 0)                  \
    macro(xmm9,  "xmm9"_s,  0, 0)                  \
    macro(xmm10, "xmm10"_s, 0, 0)                  \
    macro(xmm11, "xmm11"_s, 0, 0)                  \
    macro(xmm12, "xmm12"_s, 0, 0)                  \
    macro(xmm13, "xmm13"_s, 0, 0)                  \
    macro(xmm14, "xmm14"_s, 0, 0)                  \
    macro(xmm15, "xmm15"_s, 0, 0)

#define FOR_EACH_SP_REGISTER(macro)             \
    macro(eip,    "eip"_s,    0, 0)               \
    macro(eflags, "eflags"_s, 0, 0)

#endif // CPU(X86_64)
