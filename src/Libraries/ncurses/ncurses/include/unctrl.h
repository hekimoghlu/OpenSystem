/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

/*
 * unctrl.h
 *
 * Display a printable version of a control character.
 * Control characters are displayed in caret notation (^x), DELETE is displayed
 * as ^?. Printable characters are displayed as is.
 */

/* $Id: unctrl.h.in,v 1.11 2009/04/18 21:00:52 tom Exp $ */

#ifndef NCURSES_UNCTRL_H_incl
#define NCURSES_UNCTRL_H_incl	1

#undef  NCURSES_VERSION
#define NCURSES_VERSION "6.0"

#include <curses.h>

#ifdef __cplusplus
extern "C" {
#endif

#undef unctrl
NCURSES_EXPORT(NCURSES_CONST char *) unctrl (chtype);

#if 0
NCURSES_EXPORT(NCURSES_CONST char *) NCURSES_SP_NAME(unctrl) (SCREEN*, chtype);
#endif

#ifdef __cplusplus
}
#endif

#endif /* NCURSES_UNCTRL_H_incl */
