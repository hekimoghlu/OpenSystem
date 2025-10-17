/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include <sys/types.h>

#include <stdbool.h>
#include <stdint.h>

#define	DEBUGGING

/* constants */

#define	MAXHUNKSIZE 200000	/* is this enough lines? */
#define	INITHUNKMAX 125		/* initial dynamic allocation size */
#define	INITLINELEN 4096
#define	BUFFERSIZE 4096
#define	LINENUM_MAX LONG_MAX

#define	ORIGEXT ".orig"
#define	REJEXT ".rej"

/* handy definitions */

#define	strEQ(s1,s2) (strcmp(s1, s2) == 0)
#define	strnNE(s1,s2,l) (strncmp(s1, s2, l) != 0)
#ifdef __APPLE__
/* Binary-safe strnNE (UTF-16, internal NUL safe)*/
#define	bstrnNE(s1,s2,l) (memcmp(s1, s2, l) != 0)
#endif
#define	strnEQ(s1,s2,l) (strncmp(s1, s2, l) == 0)

/* typedefs */

typedef long    LINENUM;	/* must be signed */

/* globals */

extern mode_t	filemode;

extern char	*buf;		/* general purpose buffer */		
extern size_t	buf_size;	/* size of general purpose buffer */

extern bool	using_plan_a;	/* try to keep everything in memory */
extern bool	out_of_mem;	/* ran out of memory in plan a */
extern bool	nonempty_patchf_seen;	/* seen a non-zero-length patch file? */

#define	MAXFILEC 2

extern char	*filearg[MAXFILEC];
extern bool	ok_to_create_file;
extern char	*outname;
extern char	*origprae;

extern char	*TMPOUTNAME;
extern char	*TMPINNAME;
extern char	*TMPREJNAME;
extern char	*TMPPATNAME;
extern bool	toutkeep;
extern bool	trejkeep;

#ifdef DEBUGGING
extern int	debug;
#endif

extern bool	force;
extern bool	batch;
extern bool	verbose;
#ifdef __APPLE__
extern bool	quiet;
#endif
extern bool	reverse;
extern bool	noreverse;
extern bool	skip_rest_of_patch;
extern int	strippath;
extern bool	canonicalize;
/* TRUE if -C was specified on command line.  */
extern bool	check_only;
extern bool	warn_on_invalid_line;
extern bool	last_line_missing_eol;


#define	CONTEXT_DIFF 1
#define	NORMAL_DIFF 2
#define	ED_DIFF 3
#define	NEW_CONTEXT_DIFF 4
#define	UNI_DIFF 5

extern int	diff_type;
extern char	*revision;	/* prerequisite revision, if any */
extern LINENUM	input_lines;	/* how long is input file in lines */

extern int	posix;

#ifdef __APPLE__
enum vcsopt {
	VCS_DEFAULT,
	VCS_DISABLED,
	VCS_PROMPT,
	VCS_ALWAYS
};

enum quote_options {
	QO_LITERAL,
	QO_SHELL,
	QO_SHELL_ALWAYS,
	QO_C,
	QO_ESCAPE,	/* C without quotes */
};

extern enum vcsopt	vcsget;
extern enum quote_options	quote_opt;

int vcs_probe(const char *filename, bool missing, bool check_only);
bool vcs_prompt(const char *filename);
bool vcs_supported(void);
const char *vcs_name(void);
int vcs_checkout(const char *filename, bool missing);

extern long	settime_gmtoff;
extern bool	settime;

extern time_t	mtime_old;
extern time_t	mtime_new;
#endif
