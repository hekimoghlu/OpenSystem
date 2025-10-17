/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#ifndef _ASM_ARM_PERF_REGS_H
#define _ASM_ARM_PERF_REGS_H
enum perf_event_arm_regs {
  PERF_REG_ARM_R0,
  PERF_REG_ARM_R1,
  PERF_REG_ARM_R2,
  PERF_REG_ARM_R3,
  PERF_REG_ARM_R4,
  PERF_REG_ARM_R5,
  PERF_REG_ARM_R6,
  PERF_REG_ARM_R7,
  PERF_REG_ARM_R8,
  PERF_REG_ARM_R9,
  PERF_REG_ARM_R10,
  PERF_REG_ARM_FP,
  PERF_REG_ARM_IP,
  PERF_REG_ARM_SP,
  PERF_REG_ARM_LR,
  PERF_REG_ARM_PC,
  PERF_REG_ARM_MAX,
};
#endif
