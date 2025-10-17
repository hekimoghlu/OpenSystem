/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 ****************************************************************************/

/* $Id: termcap.h.in,v 1.16 2001/03/24 21:53:27 tom Exp $ */

#ifndef NCURSES_TERMCAP_H_incl
#define NCURSES_TERMCAP_H_incl	1

#undef  NCURSES_VERSION
#define NCURSES_VERSION "6.0"

#include <ncurses_dll.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#include <sys/types.h>

#undef  NCURSES_CONST 
#define NCURSES_CONST /*nothing*/ 

#undef  NCURSES_OSPEED 
#define NCURSES_OSPEED short 

extern NCURSES_EXPORT_VAR(char) PC;
extern NCURSES_EXPORT_VAR(char *) UP;
extern NCURSES_EXPORT_VAR(char *) BC;
extern NCURSES_EXPORT_VAR(NCURSES_OSPEED) ospeed; 

#if !defined(NCURSES_TERM_H_incl)
extern NCURSES_EXPORT(char *) tgetstr (NCURSES_CONST char *, char **);
extern NCURSES_EXPORT(char *) tgoto (const char *, int, int);
extern NCURSES_EXPORT(int) tgetent (char *, const char *);
extern NCURSES_EXPORT(int) tgetflag (NCURSES_CONST char *);
extern NCURSES_EXPORT(int) tgetnum (NCURSES_CONST char *);
extern NCURSES_EXPORT(int) tputs (const char *, int, int (*)(int));
#endif

#ifdef __cplusplus
}
#endif

#endif /* NCURSES_TERMCAP_H_incl */
