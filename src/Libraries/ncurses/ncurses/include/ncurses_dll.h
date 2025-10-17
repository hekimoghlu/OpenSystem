/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
/* $Id: ncurses_dll.h.in,v 1.9 2014/08/02 21:30:20 tom Exp $ */

#ifndef NCURSES_DLL_H_incl
#define NCURSES_DLL_H_incl 1

/* 2014-08-02 workaround for broken MinGW compiler.
 * Oddly, only TRACE is mapped to trace - the other -D's are okay.
 * suggest TDM as an alternative.
 */
#if defined(__MINGW64__)
#elif defined(__MINGW32__)
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)

#ifdef trace
#undef trace
#define TRACE
#endif

#endif	/* broken compiler */
#endif	/* MingW */

/*
 * For reentrant code, we map the various global variables into SCREEN by
 * using functions to access them.
 */
#define NCURSES_PUBLIC_VAR(name) _nc_##name
#define NCURSES_WRAPPED_VAR(type,name) extern type NCURSES_PUBLIC_VAR(name)(void)

/* no longer needed on cygwin or mingw, thanks to auto-import       */
/* but this structure may be useful at some point for an MSVC build */
/* so, for now unconditionally define the important flags           */
/* "the right way" for proper static and dll+auto-import behavior   */
#undef NCURSES_DLL
#define NCURSES_STATIC

#if defined(__CYGWIN__) || defined(__MINGW32__)
#  if defined(NCURSES_DLL)
#    if defined(NCURSES_STATIC)
#      undef NCURSES_STATIC
#    endif
#  endif
#  undef NCURSES_IMPEXP
#  undef NCURSES_API
#  undef NCURSES_EXPORT
#  undef NCURSES_EXPORT_VAR
#  if defined(NCURSES_DLL)
/* building a DLL */
#    define NCURSES_IMPEXP __declspec(dllexport)
#  elif defined(NCURSES_STATIC)
/* building or linking to a static library */
#    define NCURSES_IMPEXP /* nothing */
#  else
/* linking to the DLL */
#    define NCURSES_IMPEXP __declspec(dllimport)
#  endif
#  define NCURSES_API __cdecl
#  define NCURSES_EXPORT(type) NCURSES_IMPEXP type NCURSES_API
#  define NCURSES_EXPORT_VAR(type) NCURSES_IMPEXP type
#endif

/* Take care of non-cygwin platforms */
#if !defined(NCURSES_IMPEXP)
#  define NCURSES_IMPEXP /* nothing */
#endif
#if !defined(NCURSES_API)
#  define NCURSES_API /* nothing */
#endif
#if !defined(NCURSES_EXPORT)
#  define NCURSES_EXPORT(type) NCURSES_IMPEXP type NCURSES_API
#endif
#if !defined(NCURSES_EXPORT_VAR)
#  define NCURSES_EXPORT_VAR(type) NCURSES_IMPEXP type
#endif

#endif /* NCURSES_DLL_H_incl */
