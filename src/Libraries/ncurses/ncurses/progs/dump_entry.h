/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
 *     and: Thomas E. Dickey                        1996-on                 *
 ****************************************************************************/

/*
 * $Id: dump_entry.h,v 1.35 2015/05/27 00:56:54 tom Exp $
 *
 * Dump control definitions and variables
 */

#ifndef DUMP_ENTRY_H
#define DUMP_ENTRY_H 1

/* capability output formats */
#define F_TERMINFO	0	/* use terminfo names */
#define F_VARIABLE	1	/* use C variable names */
#define F_TERMCAP	2	/* termcap names with capability conversion */
#define F_TCONVERR	3	/* as T_TERMCAP, no skip of untranslatables */
#define F_LITERAL	4	/* like F_TERMINFO, but no smart defaults */

/* capability sort modes */
#define S_DEFAULT	0	/* sort by terminfo name (implicit) */
#define S_NOSORT	1	/* don't sort */
#define S_TERMINFO	2	/* sort by terminfo names (explicit) */
#define S_VARIABLE	3	/* sort by C variable names */
#define S_TERMCAP	4	/* sort by termcap names */

/* capability types for the comparison hook */
#define CMP_BOOLEAN	0	/* comparison on booleans */
#define CMP_NUMBER	1	/* comparison on numerics */
#define CMP_STRING	2	/* comparison on strings */
#define CMP_USE		3	/* comparison on use capabilities */

typedef unsigned PredType;
typedef unsigned PredIdx;
typedef int (*PredFunc) (PredType, PredIdx);
typedef void (*PredHook) (PredType, PredIdx, const char *);

extern NCURSES_CONST char *nametrans(const char *);
extern bool has_params(const char *src);
extern int fmt_entry(TERMTYPE *, PredFunc, int, int, int, int);
extern int show_entry(void);
extern void compare_entry(PredHook, TERMTYPE *, bool);
extern void dump_entry(TERMTYPE *, int, int, int, PredFunc);
extern void dump_init(const char *, int, int, int, int, unsigned, bool, bool);
extern void dump_uses(const char *, bool);
extern void repair_acsc(TERMTYPE *tp);

#define FAIL	-1

#endif /* DUMP_ENTRY_H */
