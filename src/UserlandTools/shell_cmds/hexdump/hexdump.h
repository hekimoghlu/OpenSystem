/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include <wchar.h>

typedef struct _pr {
	struct _pr *nextpr;		/* next print unit */
#define	F_ADDRESS	0x001		/* print offset */
#define	F_BPAD		0x002		/* blank pad */
#define	F_C		0x004		/* %_c */
#define	F_CHAR		0x008		/* %c */
#define	F_DBL		0x010		/* %[EefGf] */
#define	F_INT		0x020		/* %[di] */
#define	F_P		0x040		/* %_p */
#define	F_STR		0x080		/* %s */
#define	F_U		0x100		/* %_u */
#define	F_UINT		0x200		/* %[ouXx] */
#define	F_TEXT		0x400		/* no conversions */
	u_int flags;			/* flag values */
	int bcnt;			/* byte count */
	char *cchar;			/* conversion character */
	char *fmt;			/* printf format */
	char *nospace;			/* no whitespace version */
	int mbleft;			/* bytes left of multibyte char. */
	mbstate_t mbstate;		/* conversion state */
} PR;

typedef struct _fu {
	struct _fu *nextfu;		/* next format unit */
	struct _pr *nextpr;		/* next print unit */
#define	F_IGNORE	0x01		/* %_A */
#define	F_SETREP	0x02		/* rep count set, not default */
	u_int flags;			/* flag values */
	int reps;			/* repetition count */
	int bcnt;			/* byte count */
	char *fmt;			/* format string */
} FU;

typedef struct _fs {			/* format strings */
	struct _fs *nextfs;		/* linked list of format strings */
	struct _fu *nextfu;		/* linked list of format units */
	int bcnt;
} FS;

extern FS *fshead;			/* head of format strings list */
extern FU *endfu;			/* format at end-of-data */
extern int blocksize;			/* data block size */
extern int exitval;			/* final exit value */
extern int odmode;			/* are we acting as od(1)? */
extern int length;			/* amount of data to read */
extern off_t skip;			/* amount of data to skip at start */
enum _vflag { ALL, DUP, FIRST, WAIT };	/* -v values */
extern enum _vflag vflag;

void	 add(const char *);
void	 addfile(const char *);
void	 badcnt(const char *);
void	 badconv(const char *);
void	 badfmt(const char *);
void	 badnoconv(void);
void	 badsfmt(void);
void	 bpad(PR *);
void	 conv_c(PR *, u_char *, size_t);
void	 conv_u(PR *, u_char *);
void	 display(void);
void	 doskip(const char *, int);
void	 escape(char *);
u_char	*get(void);
void	 newsyntax(int, char ***);
int	 next(char **);
void	 nomem(void);
void	 oldsyntax(int, char ***);
size_t	 peek(u_char *, size_t);
void	 rewrite(FS *);
int	 size(FS *);
void	 usage(void);
