/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#ifndef	_IODBC_H
#define _IODBC_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef VERSION
#define VERSION		"3.52.6"
#define MAJ_VERSION	"3"
#define MIN_VERSION	"52"
#endif

#ifndef IODBC_BUILD
#define IODBC_BUILD 6071008	/* 0001.0928 */
#endif

#if	!defined(WINDOWS) && !defined(WIN32_SYSTEM)
#define _UNIX_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#define MEM_ALLOC(size)	(malloc((size_t)(size)))
#define MEM_FREE(ptr)	{if(ptr) free(ptr);}

#define STRCPY(t, s)	(strcpy((char*)(t), (char*)(s)))
#define STRNCPY(t,s,n)	(strncpy((char*)(t), (char*)(s), (size_t)(n)))
#define STRCAT(t, s)	(strcat((char*)(t), (char*)(s)))
#define STRNCAT(t,s,n)	(strncat((char*)(t), (char*)(s), (size_t)(n)))
#define STREQ(a, b)	(strcmp((char*)(a), (char*)(b)) == 0)
#define STRNEQ(a, b, n)		(strncmp((char*)(a), (char*)(b), (size_t)(n)) == 0)
#define STRLEN(str)	((str)? strlen((char*)(str)):0)
#define STRDUP(t)		(strdup((char*)(t)))
#define STRCASEEQ(a, b)		(strcasecmp((char*)(a), (char*)(b)) == 0)
#define STRNCASEEQ(a, b, n)	(strncasecmp((char*)(a), (char*)(b), (size_t)(n)) == 0)

#define WCSCPY(t, s)		(wcscpy((wchar_t*)(t), (wchar_t*)(s)))
#define WCSNCPY(t,s,n)		(wcsncpy((wchar_t*)(t), (wchar_t*)(s), (size_t)(n)))
#define WCSCAT(t, s)		(wcscat((wchar_t*)(t), (wchar_t*)(s)))
#define WCSNCAT(t,s,n)		(wcsncat((wchar_t*)(t), (wchar_t*)(s), (size_t)(n)))
#define WCSEQ(a, b)		(wcscmp((wchar_t*)(a), (wchar_t*)(b)) == 0)
#define WCSNEQ(a, b, n)		(wcsncmp((wchar_t*)(a), (wchar_t*)(b), (size_t)(n)) == 0)
#define WCSLEN(str)		((str)? wcslen((wchar_t*)(str)):0)
#define WCSDUP(t)		(wcsdup((wchar_t*)(t)))
#define WCSCASEEQ(a, b)		(wcscasecmp((wchar_t*)(a), (wchar_t*)(b)) == 0)
#define WCSNCASEEQ(a, b, n)	(wcsncasecmp((wchar_t*)(a), (wchar_t*)(b), (size_t)(n)) == 0)


#define EXPORT
#define CALLBACK
#define FAR

#ifndef WIN32
#define UNALIGNED
#endif

/*
 *  If not defined, use this as the system default odbc.ini file
 */
#if !defined(SYS_ODBC_INI) || (defined(__APPLE__) && !defined(ODBC_INI_APP))
# if defined(__BEOS__)
# 	define SYS_ODBC_INI "/boot/beos/etc/odbc.ini"
# elif defined(_MAC)
# 	ifdef __POWERPC__
# 		define SYS_ODBC_INI "Boot:System Folder:Preferences:ODBC Preferences PPC"
# 	else
# 		define SYS_ODBC_INI "Boot:System Folder:Preferences:ODBC Preferences"
# 	endif
# elif defined(__APPLE__)
# 	define SYS_ODBC_INI "/etc/odbc.ini"
# 	define ODBC_INI_APP "/Library/ODBC/odbc.ini"
# else
# 	define SYS_ODBC_INI "/etc/odbc.ini"
# endif
#endif

#if !defined(SYS_ODBCINST_INI) || (defined(__APPLE__) && !defined(ODBCINST_INI_APP))
#  if defined(__BEOS__)
#    define SYS_ODBCINST_INI	"/boot/beos/etc/odbcinst.ini"
#  elif defined(macintosh)
#  elif defined(__APPLE__)
#    define SYS_ODBCINST_INI	"/etc/odbcinst.ini"
#    define ODBCINST_INI_APP	"/Library/ODBC/odbcinst.ini"
#  else
#    define SYS_ODBCINST_INI	"/etc/odbcinst.ini"
#  endif
#endif

#endif /* _UNIX_ */

#if	defined(WINDOWS) || defined(WIN32_SYSTEM)
#include <windows.h>
#include <windowsx.h>

#ifdef	_MSVC_
#define MEM_ALLOC(size)	(fmalloc((size_t)(size)))
#define MEM_FREE(ptr)	((ptr)? ffree((PTR)(ptr)):0)
#define STRCPY(t, s)	(fstrcpy((char FAR*)(t), (char FAR*)(s)))
#define STRNCPY(t,s,n)	(fstrncpy((char FAR*)(t), (char FAR*)(s), (size_t)(n)))
#define STRLEN(str)	((str)? fstrlen((char FAR*)(str)):0)
#define STREQ(a, b)	(fstrcmp((char FAR*)(a), (char FAR*)(b) == 0))
#define STRCAT(t, s)	(strcat((char*)(t), (char*)(s)))
#define STRNCAT(t,s,n)	(strncat((char*)(t), (char*)(s), (size_t)(n)))
#define STRNCMP(t,s,n)	(strncmp((char*)(t), (char*)(s), (size_t)(n)))
#endif

#ifdef	_BORLAND_
#define MEM_ALLOC(size)	(farmalloc((unsigned long)(size)))
#define MEM_FREE(ptr)	((ptr)? farfree((void far*)(ptr)):0)
#define STRCPY(t, s)	(_fstrcpy((char FAR*)(t), (char FAR*)(s)))
#define STRNCPY(t,s,n)	(_fstrncpy((char FAR*)(t), (char FAR*)(s), (size_t)(n)))
#define STRLEN(str)     ((str)? _fstrlen((char FAR*)(str)):0)
#define STREQ(a, b)     (_fstrcmp((char FAR*)(a), (char FAR*)(b) == 0))
#define STRCAT(t, s)	(strcat((char*)(t), (char*)(s)))
#define STRNCAT(t,s,n)	(strncat((char*)(t), (char*)(s), (size_t)(n)))
#define STRNCMP(t,s,n)	(strncmp((char*)(t), (char*)(s), (size_t)(n)))
#endif

#endif /* WINDOWS */

#ifdef VMS
/*
 *  VMS also defines _UNIX_ above. This is OK for iODBC since all used UNIX
 *  interfaces are supported.
 *  The DEC C RTL actually supports dlopen(), etc, but I have made my own
 *  implementation that supports:
 *     - Proper error messages from dlopen()
 *     - The ability to place the driver in other directories than SYS$SHARE:
 *     - Neither implementation can do dlopen(NULL,), but my implementation
 *       will not crash in this case.
 *  To use old DEC C dlopen() implementation, remove the following define.
 */
#define DLDAPI_VMS_IODBC	/* Use enhanced dlopen() */
#endif

#define SYSERR		(-1)

#ifndef	NULL
#define NULL		((void *)0UL)
#endif

/*
 *  Map generic pointer to internal pointer 
 */
#define STMT(stmt, var) \
	STMT_t *stmt = (STMT_t *)var

#define CONN(con, var) \
	DBC_t *con = (DBC_t *)var

#define GENV(genv, var) \
	GENV_t *genv = (GENV_t *)var

#define ENVR(env, var) \
	ENV_t *env = (ENV_t *)var

#define DESC(desc, var) \
	DESC_t *desc = (DESC_t *)var

#define NEW_VAR(type, var) \
	type *var = (type *)MEM_ALLOC(sizeof(type))


/* these are deprecated defines from the odbc headers */
#define SQL_CONNECT_OPT_DRVR_START      1000

#endif /* _IODBC_H */
