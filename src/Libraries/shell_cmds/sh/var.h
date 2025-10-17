/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
/*
 * Shell variables.
 */

/* flags */
#define VEXPORT		0x01	/* variable is exported */
#define VREADONLY	0x02	/* variable cannot be modified */
#define VSTRFIXED	0x04	/* variable struct is statically allocated */
#define VTEXTFIXED	0x08	/* text is statically allocated */
#define VSTACK		0x10	/* text is allocated on the stack */
#define VUNSET		0x20	/* the variable is not set */
#define VNOFUNC		0x40	/* don't call the callback function */
#define VNOSET		0x80	/* do not set variable - just readonly test */
#define VNOLOCAL	0x100	/* ignore forcelocal */


struct var {
	struct var *next;		/* next entry in hash list */
	int flags;			/* flags are defined above */
	int name_len;			/* length of name */
	char *text;			/* name=value */
	void (*func)(const char *);
					/* function to be called when  */
					/* the variable gets set/unset */
};


struct localvar {
	struct localvar *next;		/* next local variable in list */
	struct var *vp;			/* the variable that was made local */
	int flags;			/* saved flags */
	char *text;			/* saved text */
};


extern struct localvar *localvars;
extern int forcelocal;

extern struct var vifs;
extern struct var vmail;
extern struct var vmpath;
extern struct var vpath;
extern struct var vps1;
extern struct var vps2;
extern struct var vps4;
extern struct var vdisvfork;
#ifndef NO_HISTORY
extern struct var vhistsize;
extern struct var vterm;
#endif

extern int localeisutf8;
/* The parser uses the locale that was in effect at startup. */
extern int initial_localeisutf8;

/*
 * The following macros access the values of the above variables.
 * They have to skip over the name.  They return the null string
 * for unset variables.
 */

#define ifsval()	(vifs.text + 4)
#define ifsset()	((vifs.flags & VUNSET) == 0)
#define mailval()	(vmail.text + 5)
#define mpathval()	(vmpath.text + 9)
#define pathval()	(vpath.text + 5)
#define ps1val()	(vps1.text + 4)
#define ps2val()	(vps2.text + 4)
#define ps4val()	(vps4.text + 4)
#define optindval()	(voptind.text + 7)
#ifndef NO_HISTORY
#define histsizeval()	(vhistsize.text + 9)
#define termval()	(vterm.text + 5)
#endif

#define mpathset()	((vmpath.flags & VUNSET) == 0)
#define disvforkset()	((vdisvfork.flags & VUNSET) == 0)

void initvar(void);
void setvar(const char *, const char *, int);
void setvareq(char *, int);
struct arglist;
void listsetvar(struct arglist *, int);
char *lookupvar(const char *);
char *bltinlookup(const char *, int);
void bltinsetlocale(void);
void bltinunsetlocale(void);
void updatecharset(void);
void initcharset(void);
char **environment(void);
int showvarscmd(int, char **);
void mklocal(char *);
void poplocalvars(void);
int unsetvar(const char *);
int setvarsafe(const char *, const char *, int);
