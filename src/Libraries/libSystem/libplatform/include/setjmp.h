/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#ifndef _BSD_SETJMP_H
#define _BSD_SETJMP_H

#include <sys/cdefs.h>
#include <Availability.h>

#if defined(__x86_64__)
/*
 * _JBLEN is number of ints required to save the following:
 * rflags, rip, rbp, rsp, rbx, r12, r13, r14, r15... these are 8 bytes each
 * mxcsr, fp control word, sigmask... these are 4 bytes each
 * add 16 ints for future expansion needs...
 */
#define _JBLEN ((9 * 2) + 3 + 16)
typedef int jmp_buf[_JBLEN];
typedef int sigjmp_buf[_JBLEN + 1];

#elif defined(__i386__)

/*
 * _JBLEN is number of ints required to save the following:
 * eax, ebx, ecx, edx, edi, esi, ebp, esp, ss, eflags, eip,
 * cs, de, es, fs, gs == 16 ints
 * onstack, mask = 2 ints
 */

#define _JBLEN (18)
typedef int jmp_buf[_JBLEN];
typedef int sigjmp_buf[_JBLEN + 1];

#elif defined(__arm__) && !defined(__ARM_ARCH_7K__)

#include <machine/signal.h>

/*
 *	_JBLEN is number of ints required to save the following:
 *	r4-r8, r10, fp, sp, lr, sig  == 10 register_t sized
 *	s16-s31 == 16 register_t sized + 1 int for FSTMX
 *	1 extra int for future use
 */
#define _JBLEN		(10 + 16 + 2)
#define _JBLEN_MAX	_JBLEN

typedef int jmp_buf[_JBLEN];
typedef int sigjmp_buf[_JBLEN + 1];

#elif defined(__arm64__) || defined(__ARM_ARCH_7K__)
/*
 * _JBLEN is the number of ints required to save the following:
 * r21-r29, sp, fp, lr == 12 registers, 8 bytes each. d8-d15
 * are another 8 registers, each 8 bytes long. (aapcs64 specifies
 * that only 64-bit versions of FP registers need to be saved).
 * Finally, two 8-byte fields for signal handling purposes.
 */
#define _JBLEN		((14 + 8 + 2) * 2)

typedef int jmp_buf[_JBLEN];
typedef int sigjmp_buf[_JBLEN + 1];

#else
#	error Undefined platform for setjmp
#endif

__BEGIN_DECLS
extern int	setjmp(jmp_buf);
extern void longjmp(jmp_buf, int) __dead2;

#ifndef _ANSI_SOURCE
int	_setjmp(jmp_buf);
void	_longjmp(jmp_buf, int) __dead2;
int	sigsetjmp(sigjmp_buf, int);
void	siglongjmp(sigjmp_buf, int) __dead2;
#endif /* _ANSI_SOURCE  */

#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
void	longjmperror(void);
#endif /* neither ANSI nor POSIX */
__END_DECLS

#endif /* _BSD_SETJMP_H */
