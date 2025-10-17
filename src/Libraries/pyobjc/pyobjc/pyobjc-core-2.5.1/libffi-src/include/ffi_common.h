/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#ifndef FFI_COMMON_H
#define FFI_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fficonfig.h"

/*	Do not move this. Some versions of AIX are very picky about where
	this is positioned. */
#ifdef __GNUC__
#	define alloca __builtin_alloca
#else
#	if HAVE_ALLOCA_H
#		include <alloca.h>
#	else
#		ifdef _AIX
#			pragma alloca
#		else
#			ifndef alloca	/* predefined by HP cc +Olibcalls */
char* alloca();
#			endif
#		endif
#	endif
#endif

/*	Check for the existence of memcpy. */
#if STDC_HEADERS
#	include <string.h>
#else
#	ifndef HAVE_MEMCPY
#		define memcpy(d, s, n) bcopy((s), (d), (n))
#	endif
#endif

#ifdef FFI_DEBUG
#include <stdio.h>

/*@exits@*/ void
ffi_assert(
/*@temp@*/	char*	expr,
/*@temp@*/	char*	file,
			int		line);
void
ffi_stop_here(void);
void
ffi_type_test(
/*@temp@*/ /*@out@*/	ffi_type*	a,
/*@temp@*/				char*	file,
						int		line);

#	define FFI_ASSERT(x)			((x) ? (void)0 : ffi_assert(#x, __FILE__,__LINE__))
#	define FFI_ASSERT_AT(x, f, l)	((x) ? 0 : ffi_assert(#x, (f), (l)))
#	define FFI_ASSERT_VALID_TYPE(x)	ffi_type_test(x, __FILE__, __LINE__)
#else
#	define FFI_ASSERT(x) 
#	define FFI_ASSERT_AT(x, f, l)
#	define FFI_ASSERT_VALID_TYPE(x)
#endif	// #ifdef FFI_DEBUG

#define ALIGN(v, a)	(((size_t)(v) + (a) - 1) & ~((a) - 1))

/*	Perform machine dependent cif processing */
ffi_status
ffi_prep_cif_machdep(
	ffi_cif*	cif);

/*	Extended cif, used in callback from assembly routine */
typedef struct	extended_cif {
/*@dependent@*/	ffi_cif*	cif;
/*@dependent@*/	void*		rvalue;
/*@dependent@*/	void**		avalue;
} extended_cif;

/*	Terse sized type definitions.  */
typedef unsigned int	UINT8	__attribute__((__mode__(__QI__)));
typedef signed int		SINT8	__attribute__((__mode__(__QI__)));
typedef unsigned int	UINT16	__attribute__((__mode__(__HI__)));
typedef signed int		SINT16	__attribute__((__mode__(__HI__)));
typedef unsigned int	UINT32	__attribute__((__mode__(__SI__)));
typedef signed int		SINT32	__attribute__((__mode__(__SI__)));
typedef unsigned int	UINT64	__attribute__((__mode__(__DI__)));
typedef signed int		SINT64	__attribute__((__mode__(__DI__)));
typedef float			FLOAT32;

#ifdef __cplusplus
}
#endif

#endif	// #ifndef FFI_COMMON_H