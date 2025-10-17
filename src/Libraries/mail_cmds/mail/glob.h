/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
 * A bunch of global variable declarations lie herein.
 * def.h must be included first.
 */

extern int	msgCount;			/* Count of messages read in */
extern int	rcvmode;			/* True if receiving mail */
extern int	sawcom;				/* Set after first command */
extern int	senderr;			/* An error while checking */
extern int	edit;				/* Indicates editing a file */
extern int	readonly;			/* Will be unable to rewrite file */
extern int	noreset;			/* String resets suspended */
extern int	sourcing;			/* Currently reading variant file */
extern int	loading;			/* Loading user definitions */
extern int	cond;				/* Current state of conditional exc. */
extern FILE	*itf;				/* Input temp file buffer */
extern FILE	*otf;				/* Output temp file buffer */
extern int	image;				/* File descriptor for image of msg */
extern FILE	*input;				/* Current command input file */
extern char	mailname[PATHSIZE];		/* Name of current file */
extern char	prevfile[PATHSIZE];		/* Name of previous file */
extern char	*homedir;			/* Path name of home directory */
extern char	*myname;			/* My login name */
extern off_t	mailsize;			/* Size of system mailbox */
extern int	lexnumber;			/* Number of TNUMBER from scan() */
extern char	lexstring[STRINGLEN];		/* String from TSTRING, scan() */
extern int	regretp;			/* Pointer to TOS of regret tokens */
extern int	regretstack[REGDEP];		/* Stack of regretted tokens */
extern char	*string_stack[REGDEP];		/* Stack of regretted strings */
extern int	numberstack[REGDEP];		/* Stack of regretted numbers */
extern struct	message	*dot;			/* Pointer to current message */
extern struct	message	*message;		/* The actual message structure */
extern struct	var	*variables[HSHSIZE];	/* Pointer to active var list */
extern struct	grouphead	*groups[HSHSIZE];/* Pointer to active groups */
extern struct	ignoretab	ignore[2];	/* ignored and retained fields
					   0 is ignore, 1 is retain */
extern struct	ignoretab	saveignore[2];	/* ignored and retained fields
					   on save to folder */
extern struct	ignoretab	ignoreall[2];	/* special, ignore all headers */
extern char	**altnames;			/* List of alternate names for user */
extern int	debug;				/* Debug flag set */
extern int	screenwidth;			/* Screen width, or best guess */
extern int	screenheight;			/* Screen height, or best guess,
					   for "header" command */
extern int	realscreenheight;		/* the real screen height */

#include <setjmp.h>

extern jmp_buf	srbuf;


/*
 * The pointers for the string allocation routines,
 * there are NSPACE independent areas.
 * The first holds STRINGSIZE bytes, the next
 * twice as much, and so on.
 */

#define	NSPACE	25			/* Total number of string spaces */
extern struct strings {
	char	*s_topFree;		/* Beginning of this area */
	char	*s_nextFree;		/* Next alloctable place here */
	unsigned s_nleft;		/* Number of bytes left here */
} stringdope[NSPACE];
