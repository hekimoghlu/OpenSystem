/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#ifndef _IODBCUNIX_H
#define _IODBCUNIX_H

/*
 *  Standard header files
 */
#include <stdlib.h>
#include <unistd.h>
#include <objc/objc.h>

#if defined(__WCHAR_TYPE__) && !defined(MACOSX102)
#include <wchar.h>
#endif


/*
 *  Windows-style declarations
 */
#define NEAR
#define FAR
#define EXPORT
#define PASCAL
#define VOID			void
#define CALLBACK
#define _cdecl
#define __stdcall


/*
 *  Boolean support
 */
#ifndef TRUE
#define TRUE			1
#endif
#ifndef FALSE
#define FALSE			0
#endif


#ifdef __cplusplus
extern "C" {
#endif


/*
 *  Windows-style typedefs
 */
#if defined (OBSOLETE_WINDOWS_TYPES)
typedef unsigned char		BYTE;
#endif
typedef unsigned short		WORD;
typedef unsigned int		DWORD;
typedef char *			LPSTR;
typedef const char *		LPCSTR;
typedef wchar_t *		LPWSTR;
typedef const wchar_t *		LPCWSTR;
typedef DWORD *			LPDWORD;

#if !defined(BOOL) && !defined(_OBJC_OBJC_H_)
typedef int			BOOL;
#endif


/*
 *  Determine sizeof(long) in case it is not determined by configure/config.h
 */
#ifndef SIZEOF_LONG
#if defined (_LP64)	|| \
    defined (__LP64__)	|| \
    defined (__64BIT__)	|| \
    defined (__alpha)	|| \
    defined (__sparcv9)	|| \
    defined (__arch64__)
#define SIZEOF_LONG	8		/* 64 bit operating systems */
#else
#define SIZEOF_LONG	4		/* 32 bit operating systems */
#endif
#endif /* SIZEOF_LONG */

#ifdef __cplusplus
}
#endif

#endif /* _IODBCUNIX_H */
