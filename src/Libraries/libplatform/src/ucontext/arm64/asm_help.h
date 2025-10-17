/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
/* ASM Macro helpers */
#if defined(__ASSEMBLER__)

.macro ARM64_STACK_PROLOG
#if __has_feature(ptrauth_returns)
	pacibsp
#endif
.endmacro

.macro ARM64_STACK_EPILOG
#if __has_feature(ptrauth_returns)
	retab
#else
	ret
#endif
.endmacro

#define PUSH_FRAME			\
	stp fp, lr, [sp, #-16]!		%% \
	mov fp, sp			%%

#define POP_FRAME			\
	mov sp, fp			%% \
	ldp fp, lr, [sp], #16		%%
#endif /* ASSEMBLER */

/* Offsets of the various register states inside of the mcontext data */
#define MCONTEXT_OFFSET_X0 16

#define MCONTEXT_OFFSET_X19_X20 168
#define MCONTEXT_OFFSET_X21_X22 184
#define MCONTEXT_OFFSET_X23_X24 200

#define MCONTEXT_OFFSET_X25_X26 216
#define MCONTEXT_OFFSET_X27_X28 232

#define MCONTEXT_OFFSET_FP_LR 248
#define MCONTEXT_OFFSET_SP 264
#define MCONTEXT_OFFSET_FLAGS 284

#define MCONTEXT_OFFSET_D8 424
#define MCONTEXT_OFFSET_D9 440
#define MCONTEXT_OFFSET_D10 456
#define MCONTEXT_OFFSET_D11 472
#define MCONTEXT_OFFSET_D12 488
#define MCONTEXT_OFFSET_D13 504
#define MCONTEXT_OFFSET_D14 520
#define MCONTEXT_OFFSET_D15 536

#if __has_feature(ptrauth_calls)
#define LR_SIGNED_WITH_IB 0x2 /* Copied from __DARWIN_ARM_THREAD_STATE64_FLAGS_IB_SIGNED_LR */
#define LR_SIGNED_WITH_IB_BIT 0x1
#endif
