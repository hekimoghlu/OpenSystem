/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#if !defined __X86INTRIN_H && !defined __IMMINTRIN_H
#error "Never use <movdirintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef _MOVDIRINTRIN_H
#define _MOVDIRINTRIN_H

/* Move doubleword as direct store */
static __inline__ void
__attribute__((__always_inline__, __nodebug__,  __target__("movdiri")))
_directstoreu_u32 (void *__dst, unsigned int  __value)
{
  __builtin_ia32_directstore_u32((unsigned int *)__dst, (unsigned int)__value);
}

#ifdef __x86_64__

/* Move quadword as direct store */
static __inline__ void
__attribute__((__always_inline__, __nodebug__,  __target__("movdiri")))
_directstoreu_u64 (void *__dst, unsigned long __value)
{
  __builtin_ia32_directstore_u64((unsigned long *)__dst, __value);
}

#endif /* __x86_64__ */

/*
 * movdir64b - Move 64 bytes as direct store.
 * The destination must be 64 byte aligned, and the store is atomic.
 * The source address has no alignment requirement, and the load from
 * the source address is not atomic.
 */
static __inline__ void
__attribute__((__always_inline__, __nodebug__,  __target__("movdir64b")))
_movdir64b (void *__dst __attribute__((align_value(64))), const void *__src)
{
  __builtin_ia32_movdir64b(__dst, __src);
}

#endif /* _MOVDIRINTRIN_H */
