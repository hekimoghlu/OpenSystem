/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
/*
 * @OSF_COPYRIGHT@
 */
/*
 * @APPLE_FREE_COPYRIGHT@
 */
/*
 *	File:		rtclock_asm_native.h
 *	Purpose:	Native routines for reading nanotime
 */

#ifndef _PAL_RTCLOCK_ASM_NATIVE_H_
#define _PAL_RTCLOCK_ASM_NATIVE_H_

/*
 * Assembly snippet included in exception handlers and rtc_nanotime_read()
 *
 *
 * Warning!  There are several copies of this code in the trampolines found in
 * osfmk/x86_64/idt64.s, coming from the various TIMER macros in rtclock_asm.h.
 * They're all kept in sync by using the RTC_NANOTIME_READ() macro.
 *
 * The algorithm we use is:
 *
 *	ns = ((((rdtsc - rnt_tsc_base)<<rnt_shift)*rnt_tsc_scale) / 2**32) + rnt_ns_base;
 *
 * rnt_shift, a constant computed during initialization, is the smallest value for which:
 *
 *	(tscFreq << rnt_shift) > SLOW_TSC_THRESHOLD
 *
 * Where SLOW_TSC_THRESHOLD is about 10e9.  Since most processor's tscFreqs are greater
 * than 1GHz, rnt_shift is usually 0.  rnt_tsc_scale is also a 32-bit constant:
 *
 *	rnt_tsc_scale = (10e9 * 2**32) / (tscFreq << rnt_shift);
 *
 * %rdi points to nanotime info struct.
 * %rax returns nanotime
 */
#define PAL_RTC_NANOTIME_READ_FAST()					  \
0:	movl	RNT_GENERATION(%rdi),%esi				; \
	test        %esi,%esi		/* info updating? */		; \
        jz        0b			/* - wait if so */		; \
	lfence								; \
	rdtsc								; \
	shlq	$32,%rdx						; \
	movl    RNT_SHIFT(%rdi),%ecx					; \
	orq	%rdx,%rax			/* %rax := tsc */	; \
	subq	RNT_TSC_BASE(%rdi),%rax		/* tsc - tsc_base */	; \
	shlq    %cl,%rax						; \
	movl	RNT_SCALE(%rdi),%ecx					; \
	mulq	%rcx				/* delta * scale */	; \
	shrdq	$32,%rdx,%rax			/* %rdx:%rax >>= 32 */	; \
	addq	RNT_NS_BASE(%rdi),%rax		/* add ns_base */	; \
	cmpl	RNT_GENERATION(%rdi),%esi	/* repeat if changed */ ; \
	jne	0b

#define PAL_RTC_NANOTIME_READ_NOBARRIER()					  \
0:	movl	RNT_GENERATION(%rdi),%esi				; \
	test        %esi,%esi		/* info updating? */		; \
        jz        0b			/* - wait if so */		; \
	rdtsc								; \
	shlq	$32,%rdx						; \
	movl    RNT_SHIFT(%rdi),%ecx					; \
	orq	%rdx,%rax			/* %rax := tsc */	; \
	subq	RNT_TSC_BASE(%rdi),%rax		/* tsc - tsc_base */	; \
	shlq    %cl,%rax						; \
	movl	RNT_SCALE(%rdi),%ecx					; \
	mulq	%rcx				/* delta * scale */	; \
	shrdq	$32,%rdx,%rax			/* %rdx:%rax >>= 32 */	; \
	addq	RNT_NS_BASE(%rdi),%rax		/* add ns_base */	; \
	cmpl	RNT_GENERATION(%rdi),%esi	/* repeat if changed */ ; \
	jne	0b

#endif /* _PAL_RTCLOCK_ASM_NATIVE_H_ */
