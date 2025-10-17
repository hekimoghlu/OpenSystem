/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#ifndef _ASMARM_SIGCONTEXT_H
#define _ASMARM_SIGCONTEXT_H
struct sigcontext {
  unsigned long trap_no;
  unsigned long error_code;
  unsigned long oldmask;
  unsigned long arm_r0;
  unsigned long arm_r1;
  unsigned long arm_r2;
  unsigned long arm_r3;
  unsigned long arm_r4;
  unsigned long arm_r5;
  unsigned long arm_r6;
  unsigned long arm_r7;
  unsigned long arm_r8;
  unsigned long arm_r9;
  unsigned long arm_r10;
  unsigned long arm_fp;
  unsigned long arm_ip;
  unsigned long arm_sp;
  unsigned long arm_lr;
  unsigned long arm_pc;
  unsigned long arm_cpsr;
  unsigned long fault_address;
};
#endif
