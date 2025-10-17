/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#ifndef _ASM_ARM64_PERF_REGS_H
#define _ASM_ARM64_PERF_REGS_H
enum perf_event_arm_regs {
  PERF_REG_ARM64_X0,
  PERF_REG_ARM64_X1,
  PERF_REG_ARM64_X2,
  PERF_REG_ARM64_X3,
  PERF_REG_ARM64_X4,
  PERF_REG_ARM64_X5,
  PERF_REG_ARM64_X6,
  PERF_REG_ARM64_X7,
  PERF_REG_ARM64_X8,
  PERF_REG_ARM64_X9,
  PERF_REG_ARM64_X10,
  PERF_REG_ARM64_X11,
  PERF_REG_ARM64_X12,
  PERF_REG_ARM64_X13,
  PERF_REG_ARM64_X14,
  PERF_REG_ARM64_X15,
  PERF_REG_ARM64_X16,
  PERF_REG_ARM64_X17,
  PERF_REG_ARM64_X18,
  PERF_REG_ARM64_X19,
  PERF_REG_ARM64_X20,
  PERF_REG_ARM64_X21,
  PERF_REG_ARM64_X22,
  PERF_REG_ARM64_X23,
  PERF_REG_ARM64_X24,
  PERF_REG_ARM64_X25,
  PERF_REG_ARM64_X26,
  PERF_REG_ARM64_X27,
  PERF_REG_ARM64_X28,
  PERF_REG_ARM64_X29,
  PERF_REG_ARM64_LR,
  PERF_REG_ARM64_SP,
  PERF_REG_ARM64_PC,
  PERF_REG_ARM64_MAX,
  PERF_REG_ARM64_VG = 46,
  PERF_REG_ARM64_EXTENDED_MAX
};
#define PERF_REG_EXTENDED_MASK (1ULL << PERF_REG_ARM64_VG)
#endif
