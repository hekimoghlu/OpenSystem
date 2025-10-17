/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#ifndef	_SYS_CDEFS_H_

/*
 * Compiler-dependent macros to help declare dead (non-returning) and
 * pure (no side effects) functions, and unused variables.  They are
 * null except for versions of gcc that are known to support the features
 * properly (old versions of gcc-2 supported the dead and pure features
 * in a different (wrong) way).  If we do not provide an implementation
 * for a given compiler, let the compile fail if it is told to use
 * a feature that we cannot live without.
 */
#ifdef lint
#define	__packed
#define	__aligned(x)
#define	__section(x)
#else
#define	__packed	__attribute__((__packed__))
#define	__aligned(x)	__attribute__((__aligned__(x)))
#define	__section(x)	__attribute__((__section__(x)))
#endif

#if !defined(__STRICT_ANSI__) || __STDC_VERSION__ >= 199901
#define	__LONG_LONG_SUPPORTED
#endif

/*
 * We define this here since <stddef.h>, <sys/queue.h>, and <sys/types.h>
 * require it.
 */
#define	__strong_reference(sym,aliassym)	\
	extern __typeof (sym) aliassym __attribute__ ((__alias__ (#sym)));
#define	__weak_reference(sym,alias)
#define	__warn_references(sym,msg)

#ifndef	__RCSID_SOURCE
#ifndef	NO__RCSID_SOURCE
#define	__RCSID_SOURCE(s)	__IDSTRING(__CONCAT(__rcsid_source_,__LINE__),s)
#else
#define	__RCSID_SOURCE(s)
#endif
#endif

#ifndef	__DECONST
#define	__DECONST(type, var)	((type)(uintptr_t)(const void *)(var))
#endif

#ifndef	__DEVOLATILE
#define	__DEVOLATILE(type, var)	((type)(uintptr_t)(volatile void *)(var))
#endif

#ifndef	__DEQUALIFY
#define	__DEQUALIFY(type, var)	((type)(uintptr_t)(const volatile void *)(var))
#endif

/*
 * Now include the Apple version of sys/cdefs.h
 */

#include_next <sys/cdefs.h>

/*
 * Use FreeBSD version of __CONCAT
 */
#if defined(__STDC__) || defined(__cplusplus)
#undef __CONCAT1
#undef __CONCAT
#define __CONCAT1(x,y)	x ## y
#define __CONCAT(x,y)	__CONCAT1(x,y)
#endif

/*
 * Deal with all versions of POSIX.  The ordering relative to the tests above is
 * important.
 */
#ifdef _POSIX_C_SOURCE
#if _POSIX_C_SOURCE >= 200112
#define	__POSIX_VISIBLE		200112
#define	__ISO_C_VISIBLE		1999
#elif _POSIX_C_SOURCE >= 199506
#define	__POSIX_VISIBLE		199506
#define	__ISO_C_VISIBLE		1990
#elif _POSIX_C_SOURCE >= 199309
#define	__POSIX_VISIBLE		199309
#define	__ISO_C_VISIBLE		1990
#elif _POSIX_C_SOURCE >= 199209
#define	__POSIX_VISIBLE		199209
#define	__ISO_C_VISIBLE		1990
#elif _POSIX_C_SOURCE >= 199009
#define	__POSIX_VISIBLE		199009
#define	__ISO_C_VISIBLE		1990
#else
#define	__POSIX_VISIBLE		198808
#define	__ISO_C_VISIBLE		0
#endif /* _POSIX_C_SOURCE */
#else
/*-
 * Deal with _ANSI_SOURCE:
 * If it is defined, and no other compilation environment is explicitly
 * requested, then define our internal feature-test macros to zero.  This
 * makes no difference to the preprocessor (undefined symbols in preprocessing
 * expressions are defined to have value zero), but makes it more convenient for
 * a test program to print out the values.
 *
 * If a program mistakenly defines _ANSI_SOURCE and some other macro such as
 * _POSIX_C_SOURCE, we will assume that it wants the broader compilation
 * environment (and in fact we will never get here).
 */
#if defined(_ANSI_SOURCE)	/* Hide almost everything. */
#define	__POSIX_VISIBLE		0
#define	__XSI_VISIBLE		0
#define	__BSD_VISIBLE		0
#define	__ISO_C_VISIBLE		1990
#elif defined(_C99_SOURCE)	/* Localism to specify strict C99 env. */
#define	__POSIX_VISIBLE		0
#define	__XSI_VISIBLE		0
#define	__BSD_VISIBLE		0
#define	__ISO_C_VISIBLE		1999
#else				/* Default environment: show everything. */
#define	__POSIX_VISIBLE		200112
#define	__XSI_VISIBLE		600
#define	__BSD_VISIBLE		1
#define	__ISO_C_VISIBLE		1999
#endif
#endif

#endif /* !_SYS_CDEFS_H_ */
