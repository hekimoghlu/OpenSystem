/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#ifndef _ASM_X86_PERF_REGS_H
#define _ASM_X86_PERF_REGS_H
enum perf_event_x86_regs {
  PERF_REG_X86_AX,
  PERF_REG_X86_BX,
  PERF_REG_X86_CX,
  PERF_REG_X86_DX,
  PERF_REG_X86_SI,
  PERF_REG_X86_DI,
  PERF_REG_X86_BP,
  PERF_REG_X86_SP,
  PERF_REG_X86_IP,
  PERF_REG_X86_FLAGS,
  PERF_REG_X86_CS,
  PERF_REG_X86_SS,
  PERF_REG_X86_DS,
  PERF_REG_X86_ES,
  PERF_REG_X86_FS,
  PERF_REG_X86_GS,
  PERF_REG_X86_R8,
  PERF_REG_X86_R9,
  PERF_REG_X86_R10,
  PERF_REG_X86_R11,
  PERF_REG_X86_R12,
  PERF_REG_X86_R13,
  PERF_REG_X86_R14,
  PERF_REG_X86_R15,
  PERF_REG_X86_32_MAX = PERF_REG_X86_GS + 1,
  PERF_REG_X86_64_MAX = PERF_REG_X86_R15 + 1,
  PERF_REG_X86_XMM0 = 32,
  PERF_REG_X86_XMM1 = 34,
  PERF_REG_X86_XMM2 = 36,
  PERF_REG_X86_XMM3 = 38,
  PERF_REG_X86_XMM4 = 40,
  PERF_REG_X86_XMM5 = 42,
  PERF_REG_X86_XMM6 = 44,
  PERF_REG_X86_XMM7 = 46,
  PERF_REG_X86_XMM8 = 48,
  PERF_REG_X86_XMM9 = 50,
  PERF_REG_X86_XMM10 = 52,
  PERF_REG_X86_XMM11 = 54,
  PERF_REG_X86_XMM12 = 56,
  PERF_REG_X86_XMM13 = 58,
  PERF_REG_X86_XMM14 = 60,
  PERF_REG_X86_XMM15 = 62,
  PERF_REG_X86_XMM_MAX = PERF_REG_X86_XMM15 + 2,
};
#define PERF_REG_EXTENDED_MASK (~((1ULL << PERF_REG_X86_XMM0) - 1))
#endif
