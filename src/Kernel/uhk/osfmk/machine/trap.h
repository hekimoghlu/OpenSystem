/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef _MACHINE_TRAP_H
#define _MACHINE_TRAP_H

#if defined (__i386__) || defined (__x86_64__)
#include "i386/trap.h"
#elif defined (__arm__) || defined (__arm64__)
#include "arm/trap.h"
#else
#error architecture not supported
#endif

#define ml_trap_pin_value_1(a) ({ \
	register long _a __asm__(ML_TRAP_REGISTER_1) = (long)(a);               \
                                                                                \
	__asm__ __volatile__ ("" : "+r"(_a));                                   \
})
#define ml_trap_pin_value_2(a, b) ({ \
	register long _a __asm__(ML_TRAP_REGISTER_1) = (long)(a);               \
	register long _b __asm__(ML_TRAP_REGISTER_2) = (long)(b);               \
                                                                                \
	__asm__ __volatile__ ("" : "+r"(_a), "+r"(_b));                         \
})
#define ml_trap_pin_value_3(a, b, c) ({ \
	register long _a __asm__(ML_TRAP_REGISTER_1) = (long)(a);               \
	register long _b __asm__(ML_TRAP_REGISTER_2) = (long)(b);               \
	register long _c __asm__(ML_TRAP_REGISTER_3) = (long)(c);               \
                                                                                \
	__asm__ __volatile__ ("" : "+r"(_a), "+r"(_b), "+r"(_c));               \
})

#define ml_fatal_trap_with_value(code, a)  ({ \
	ml_trap_pin_value_1(a); \
	ml_fatal_trap(code); \
})

#define ml_fatal_trap_with_value2(code, a, b)  ({ \
	ml_trap_pin_value_2(a, b); \
	ml_fatal_trap(code); \
})

#define ml_fatal_trap_with_value3(code, a, b, c)  ({ \
	ml_trap_pin_value_3(a, b, c); \
	ml_fatal_trap(code); \
})

/*
 * Used for when `e` failed a linked list safe unlinking check.
 * On optimized builds, `e`'s value will be in:
 * - %rax for Intel
 * - x8 for arm64
 * - r8 on armv7
 */
__attribute__((cold, noreturn, always_inline))
static inline void
ml_fatal_trap_invalid_list_linkage(unsigned long e)
{
	ml_fatal_trap_with_value(/* XNU_HARD_TRAP_SAFE_UNLINK */ 0xbffd, e);
}

#endif /* _MACHINE_TRAP_H */
