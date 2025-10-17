/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#ifndef VPX_VPX_PORTS_ASMDEFS_MMI_H_
#define VPX_VPX_PORTS_ASMDEFS_MMI_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#if HAVE_MMI

#if HAVE_MIPS64
#define mips_reg int64_t
#define MMI_ADDU(reg1, reg2, reg3) \
  "daddu       " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_ADDIU(reg1, reg2, immediate) \
  "daddiu      " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_ADDI(reg1, reg2, immediate) \
  "daddi       " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_SUBU(reg1, reg2, reg3) \
  "dsubu       " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_L(reg, addr, bias) \
  "ld          " #reg ",        " #bias "(" #addr ")               \n\t"

#define MMI_SRL(reg1, reg2, shift) \
  "ssrld       " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_SLL(reg1, reg2, shift) \
  "dsll        " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_MTC1(reg, fp) \
  "dmtc1       " #reg ",        " #fp "                            \n\t"

#define MMI_LI(reg, immediate) \
  "dli         " #reg ",        " #immediate "                     \n\t"

#else
#define mips_reg int32_t
#define MMI_ADDU(reg1, reg2, reg3) \
  "addu        " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_ADDIU(reg1, reg2, immediate) \
  "addiu       " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_ADDI(reg1, reg2, immediate) \
  "addi        " #reg1 ",       " #reg2 ",       " #immediate "    \n\t"

#define MMI_SUBU(reg1, reg2, reg3) \
  "subu        " #reg1 ",       " #reg2 ",       " #reg3 "         \n\t"

#define MMI_L(reg, addr, bias) \
  "lw          " #reg ",        " #bias "(" #addr ")               \n\t"

#define MMI_SRL(reg1, reg2, shift) \
  "ssrlw       " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_SLL(reg1, reg2, shift) \
  "sll         " #reg1 ",       " #reg2 ",       " #shift "        \n\t"

#define MMI_MTC1(reg, fp) \
  "mtc1        " #reg ",        " #fp "                            \n\t"

#define MMI_LI(reg, immediate) \
  "li          " #reg ",        " #immediate "                     \n\t"

#endif /* HAVE_MIPS64 */

#endif /* HAVE_MMI */

#endif  // VPX_VPX_PORTS_ASMDEFS_MMI_H_
