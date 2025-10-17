/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
 * Types of address specifications
 */
enum e_atype {
	AT_RE	    = 1,			/* Line that match RE */
	AT_LINE,				/* Specific line */
	AT_RELLINE,				/* Relative line */
	AT_LAST,				/* Last line */
};

/*
 * Format of an address
 */
struct s_addr {
	enum e_atype type;			/* Address type */
	union {
		u_long l;			/* Line number */
		regex_t *r;			/* Regular expression */
	} u;
};

/*
 * Substitution command
 */
struct s_subst {
	int n;					/* Occurrence to subst. */
	int p;					/* True if p flag */
	int icase;				/* True if I flag */
	char *wfile;				/* NULL if no wfile */
	int wfd;				/* Cached file descriptor */
	regex_t *re;				/* Regular expression */
	unsigned int maxbref;			/* Largest backreference. */
	u_long linenum;				/* Line number. */
	char *new;				/* Replacement text */
};

/*
 * Translate command.
 */
struct s_tr {
	unsigned char bytetab[256];
	struct trmulti {
		size_t fromlen;
		char from[MB_LEN_MAX];
		size_t tolen;
		char to[MB_LEN_MAX];
	} *multis;
	int nmultis;
};

/*
 * An internally compiled command.
 * Initialy, label references are stored in t, on a second pass they
 * are updated to pointers.
 */
struct s_command {
	struct s_command *next;			/* Pointer to next command */
	struct s_addr *a1, *a2;			/* Start and end address */
	u_long startline;			/* Start line number or zero */
	char *t;				/* Text for : a c i r w */
	union {
		struct s_command *c;		/* Command(s) for b t { */
		struct s_subst *s;		/* Substitute command */
		struct s_tr *y;			/* Replace command array */
		int fd;				/* File descriptor for w */
	} u;
	char code;				/* Command code */
	u_int nonsel:1;				/* True if ! */
};

/*
 * Types of command arguments recognised by the parser
 */
enum e_args {
	EMPTY,			/* d D g G h H l n N p P q x = \0 */
	TEXT,			/* a c i */
	NONSEL,			/* ! */
	GROUP,			/* { */
	ENDGROUP,		/* } */
	COMMENT,		/* # */
	BRANCH,			/* b t */
	LABEL,			/* : */
	RFILE,			/* r */
	WFILE,			/* w */
	SUBST,			/* s */
	TR			/* y */
};

/*
 * Structure containing things to append before a line is read
 */
struct s_appends {
	enum {AP_STRING, AP_FILE} type;
	char *s;
	size_t len;
};

enum e_spflag {
	APPEND,					/* Append to the contents. */
	REPLACE,				/* Replace the contents. */
};

/*
 * Structure for a space (process, hold, otherwise).
 */
typedef struct {
	char *space;		/* Current space pointer. */
	size_t len;		/* Current length. */
	int deleted;		/* If deleted. */
	int append_newline;	/* If originally terminated by \n. */
	char *back;		/* Backing memory. */
	size_t blen;		/* Backing memory length. */
} SPACE;
