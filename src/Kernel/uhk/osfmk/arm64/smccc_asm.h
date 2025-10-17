/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#ifndef _ARM64_SMCCC_ASM_H_
#define _ARM64_SMCCC_ASM_H_

#ifndef __ASSEMBLER__
#error "This header should only be used in .s files"
#endif

/*
 * SAVE_SMCCC_CLOBBERED_REGISTERS
 *
 * Saves x0-x3 to stack in preparation for an hvc/smc call.
 */

.macro  SAVE_SMCCC_CLOBBERED_REGISTERS
stp             x0, x1, [sp, #- 16]!
stp             x2, x3, [sp, #- 16]!
.endmacro

/*
 * LOAD_SMCCC_CLOBBERED_REGISTERS
 *
 * Loads x0-x3 from stack after an hvc/smc call.
 */

.macro  LOAD_SMCCC_CLOBBERED_REGISTERS
ldp             x2, x3, [sp], #16
ldp             x0, x1, [sp], #16
.endmacro

#endif /* _ARM64_SMCCC_ASM_H_ */

/* vim: set ts=4 ft=asm: */
