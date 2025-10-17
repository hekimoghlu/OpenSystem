/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
/****************************************************************************
 *  Author: Thomas E. Dickey                    1996-on                     *
 ****************************************************************************/
/* $Id: nc_alloc.h,v 1.22 2013/01/26 21:56:51 tom Exp $ */

#ifndef NC_ALLOC_included
#define NC_ALLOC_included 1
/* *INDENT-OFF* */

#ifdef __cplusplus
extern "C" {
#endif

#include <ncurses_dll.h>
#include <stddef.h>

#if defined(HAVE_LIBDMALLOC) && HAVE_LIBDMALLOC
#include <string.h>
#undef strndup		/* workaround for #define in GLIBC 2.7 */
#include <dmalloc.h>    /* Gray Watson's library */
#else
#undef  HAVE_LIBDMALLOC
#define HAVE_LIBDMALLOC 0
#endif

#if defined(HAVE_LIBDBMALLOC) && HAVE_LIBDBMALLOC
#include <dbmalloc.h>   /* Conor Cahill's library */
#else
#undef  HAVE_LIBDBMALLOC
#define HAVE_LIBDBMALLOC 0
#endif

#if defined(HAVE_LIBMPATROL) && HAVE_LIBMPATROL
#include <mpatrol.h>    /* Memory-Patrol library */
#else
#undef  HAVE_LIBMPATROL
#define HAVE_LIBMPATROL 0
#endif

#ifndef NO_LEAKS
#define NO_LEAKS 0
#endif

#if HAVE_LIBDBMALLOC || HAVE_LIBDMALLOC || NO_LEAKS
#define HAVE_NC_FREEALL 1
struct termtype;
extern NCURSES_EXPORT(void) _nc_free_and_exit(int) GCC_NORETURN;
extern NCURSES_EXPORT(void) _nc_free_tinfo(int) GCC_NORETURN;
extern NCURSES_EXPORT(void) _nc_free_tic(int) GCC_NORETURN;
extern NCURSES_EXPORT(void) _nc_free_tparm(void);
extern NCURSES_EXPORT(void) _nc_leaks_dump_entry(void);
extern NCURSES_EXPORT(void) _nc_leaks_tic(void);

#if NCURSES_SP_FUNCS
extern NCURSES_EXPORT(void) NCURSES_SP_NAME(_nc_free_and_exit)(SCREEN*, int) GCC_NORETURN;
#endif

#define ExitProgram(code) _nc_free_and_exit(code)

#endif /* NO_LEAKS, etc */

#ifndef HAVE_NC_FREEALL
#define HAVE_NC_FREEALL 0
#endif

#ifndef ExitProgram
#define ExitProgram(code) exit(code)
#endif

/* doalloc.c */
extern NCURSES_EXPORT(void *) _nc_doalloc(void *, size_t);
#if !HAVE_STRDUP
#undef strdup
#define strdup _nc_strdup
extern NCURSES_EXPORT(char *) _nc_strdup(const char *);
#endif

/* entries.c */
extern NCURSES_EXPORT(void) _nc_leaks_tinfo(void);

#define typeMalloc(type,elts) (type *)malloc((size_t)(elts)*sizeof(type))
#define typeCalloc(type,elts) (type *)calloc((size_t)(elts),sizeof(type))
#define typeRealloc(type,elts,ptr) (type *)_nc_doalloc(ptr, (size_t)(elts)*sizeof(type))

#ifdef __cplusplus
}
#endif

/* *INDENT-ON* */

#endif /* NC_ALLOC_included */
