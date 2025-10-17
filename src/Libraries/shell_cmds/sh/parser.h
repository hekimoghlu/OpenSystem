/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
/* control characters in argument strings */
#define CTLESC '\300'
#define CTLVAR '\301'
#define CTLENDVAR '\371'
#define CTLBACKQ '\372'
#define CTLQUOTE 01		/* ored with CTLBACKQ code if in quotes */
/*	CTLBACKQ | CTLQUOTE == '\373' */
#define	CTLARI	'\374'
#define	CTLENDARI '\375'
#define	CTLQUOTEMARK '\376'
#define	CTLQUOTEEND '\377' /* only for ${v+-...} */

/* variable substitution byte (follows CTLVAR) */
#define VSTYPE		0x0f	/* type of variable substitution */
#define VSNUL		0x10	/* colon--treat the empty string as unset */
#define VSLINENO	0x20	/* expansion of $LINENO, the line number \
				   follows immediately */
#define VSQUOTE		0x80	/* inside double quotes--suppress splitting */

/* values of VSTYPE field */
#define VSNORMAL	0x1		/* normal variable:  $var or ${var} */
#define VSMINUS		0x2		/* ${var-text} */
#define VSPLUS		0x3		/* ${var+text} */
#define VSQUESTION	0x4		/* ${var?message} */
#define VSASSIGN	0x5		/* ${var=text} */
#define VSTRIMLEFT	0x6		/* ${var#pattern} */
#define VSTRIMLEFTMAX	0x7		/* ${var##pattern} */
#define VSTRIMRIGHT	0x8		/* ${var%pattern} */
#define VSTRIMRIGHTMAX 	0x9		/* ${var%%pattern} */
#define VSLENGTH	0xa		/* ${#var} */
#define VSERROR		0xb		/* Syntax error, issue when expanded */


/*
 * NEOF is returned by parsecmd when it encounters an end of file.  It
 * must be distinct from NULL.
 */
#define NEOF ((union node *)-1)
extern int whichprompt;		/* 1 == PS1, 2 == PS2 */
extern const char *const parsekwd[];


union node *parsecmd(int);
union node *parsewordexp(void);
void forcealias(void);
void fixredir(union node *, const char *, int);
int goodname(const char *);
int isassignment(const char *);
char *getprompt(void *);
const char *expandstr(const char *);
