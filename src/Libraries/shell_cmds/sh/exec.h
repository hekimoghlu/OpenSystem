/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
/* values of cmdtype */
#define CMDUNKNOWN -1		/* no entry in table for command */
#define CMDNORMAL 0		/* command is an executable program */
#define CMDBUILTIN 1		/* command is a shell builtin */
#define CMDFUNCTION 2		/* command is a shell function */

/* values for typecmd_impl's third parameter */
enum {
	TYPECMD_SMALLV,		/* command -v */
	TYPECMD_BIGV,		/* command -V */
	TYPECMD_TYPE		/* type */
};

union node;
struct cmdentry {
	int cmdtype;
	union param {
		int index;
		struct funcdef *func;
	} u;
	int special;
};


/* action to find_command() */
#define DO_ERR		0x01	/* prints errors */
#define DO_NOFUNC	0x02	/* don't return shell functions, for command */

extern const char *pathopt;	/* set by padvance */
extern int exerrno;		/* last exec error */

void shellexec(char **, char **, const char *, int) __dead2;
char *padvance(const char **, const char *);
void find_command(const char *, struct cmdentry *, int, const char *);
int find_builtin(const char *, int *);
void hashcd(void);
void changepath(const char *);
void defun(const char *, union node *);
int unsetfunc(const char *);
int isfunc(const char *);
int typecmd_impl(int, char **, int, const char *);
void clearcmdentry(void);
