/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
extern char *commandname;	/* currently executing command */
extern int exitstatus;		/* exit status of last command */
extern int oexitstatus;		/* saved exit status */
extern struct arglist *cmdenviron;  /* environment for builtin command */


struct backcmd {		/* result of evalbackcmd */
	int fd;			/* file descriptor to read from */
	char *buf;		/* buffer */
	int nleft;		/* number of chars in buffer */
	struct job *jp;		/* job structure for command */
};

void reseteval(void);

/* flags in argument to evaltree/evalstring */
#define EV_EXIT 01		/* exit after evaluating tree */
#define EV_TESTED 02		/* exit status is checked; ignore -e flag */
#define EV_BACKCMD 04		/* command executing within back quotes */

void evalstring(const char *, int);
union node;	/* BLETCH for ansi C */
void evaltree(union node *, int);
void evalbackcmd(union node *, struct backcmd *);

/* in_function returns nonzero if we are currently evaluating a function */
#define in_function()	funcnest
extern int funcnest;
extern int evalskip;
extern int skipcount;

/* reasons for skipping commands (see comment on breakcmd routine) */
#define SKIPBREAK	1
#define SKIPCONT	2
#define SKIPRETURN	3
