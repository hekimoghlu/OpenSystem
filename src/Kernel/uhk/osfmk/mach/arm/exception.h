/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#ifndef _MACH_ARM_EXCEPTION_H_
#define _MACH_ARM_EXCEPTION_H_

#if defined (__arm__) || defined (__arm64__)

#define EXC_TYPES_COUNT         14      /* incl. illegal exception 0 */

#define EXC_MASK_MACHINE         0

#define EXCEPTION_CODE_MAX       2      /*  code and subcode */

#if XNU_KERNEL_PRIVATE
#if __has_feature(ptrauth_calls)
#define EXC_PTRAUTH_BIT         0x200  /* bit set if exception could have been caused by ptrauth failure */
#endif /* __has_feature(ptrauth_calls) */
#endif /* XNU_KERNEL_PRIVATE */

/*
 *	Trap numbers as defined by the hardware exception vectors.
 */

/*
 *      EXC_BAD_INSTRUCTION
 */

#define EXC_ARM_UNDEFINED       1       /* Undefined */
#define EXC_ARM_SME_DISALLOWED  2       /* Current thread state prohibits use of SME resources */

/*
 *      EXC_ARITHMETIC
 */

#define EXC_ARM_FP_UNDEFINED    0       /* Undefined Floating Point Exception */
#define EXC_ARM_FP_IO           1       /* Invalid Floating Point Operation */
#define EXC_ARM_FP_DZ           2       /* Floating Point Divide by Zero */
#define EXC_ARM_FP_OF           3       /* Floating Point Overflow */
#define EXC_ARM_FP_UF           4       /* Floating Point Underflow */
#define EXC_ARM_FP_IX           5       /* Inexact Floating Point Result */
#define EXC_ARM_FP_ID           6       /* Floating Point Denormal Input */

/*
 *      EXC_BAD_ACCESS
 *      Note: do not conflict with kern_return_t values returned by vm_fault
 */

#define EXC_ARM_DA_ALIGN        0x101   /* Alignment Fault */
#define EXC_ARM_DA_DEBUG        0x102   /* Debug (watch/break) Fault */
#define EXC_ARM_SP_ALIGN        0x103   /* SP Alignment Fault */
#define EXC_ARM_SWP             0x104   /* SWP instruction */
#define EXC_ARM_PAC_FAIL        0x105   /* PAC authentication failure */


/*
 *	EXC_BREAKPOINT
 */

#define EXC_ARM_BREAKPOINT      1       /* breakpoint trap */

#endif /* defined (__arm__) || defined (__arm64__) */

#endif  /* _MACH_ARM_EXCEPTION_H_ */
