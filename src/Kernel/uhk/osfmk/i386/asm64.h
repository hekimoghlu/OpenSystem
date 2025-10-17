/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
/* Helper macros for 64-bit mode switching */


/*
 * Long jump to 64-bit space from 32-bit compatibility mode.
 * Effected, in fact, by a long return ..
 *  - we push the 64-bit kernel code selector KERNEL64_CS
 *  - call .+1 to get EIP on stack
 *  - adjust return address after lret
 *  - lret to return to next instruction but 64-bit mode.
 */
#define	ENTER_64BIT_MODE()			\
	push	$KERNEL64_CS			;\
	call    1f				;\
1:	addl    $(2f-1b), (%esp)		;\
	lret					;\
2:	.code64

/*
 * Long jump to 32-bit compatibility mode from 64-bit space.
 * Effected by long return similar to ENTER_64BIT_MODE.
 */
#define ENTER_COMPAT_MODE()			\
	call	3f				;\
3:	addq	$(4f-3b), (%rsp)		;\
	movl	$KERNEL32_CS, 4(%rsp)		;\
	lret					;\
4:	.code32

