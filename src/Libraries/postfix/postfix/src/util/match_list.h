/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#ifndef _MATCH_LIST_H_INCLUDED_
#define _MATCH_LIST_H_INCLUDED_

/*++
/* NAME
/*	match_list 3h
/* SUMMARY
/*	generic list-based pattern matching
/* SYNOPSIS
/*	#include <match_list.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <argv.h>
#include <vstring.h>

 /*
  * External interface.
  */
typedef struct MATCH_LIST MATCH_LIST;

typedef int (*MATCH_LIST_FN) (MATCH_LIST *, const char *, const char *);

struct MATCH_LIST {
    char   *pname;			/* used in error messages */
    int     flags;			/* processing options */
    ARGV   *patterns;			/* one pattern each */
    int     match_count;		/* match function/argument count */
    MATCH_LIST_FN *match_func;		/* match functions */
    const char **match_args;		/* match arguments */
    VSTRING *fold_buf;			/* case-folded pattern string */
    int     error;			/* last operation */
};

#define MATCH_FLAG_NONE		0
#define MATCH_FLAG_PARENT	(1<<0)
#define MATCH_FLAG_RETURN	(1<<1)
#define MATCH_FLAG_ALL		(MATCH_FLAG_PARENT | MATCH_FLAG_RETURN)

extern MATCH_LIST *match_list_init(const char *, int, const char *, int,...);
extern int match_list_match(MATCH_LIST *,...);
extern void match_list_free(MATCH_LIST *);

 /*
  * The following functions are not part of the public interface. These
  * functions may be called only through match_list_match().
  */
extern int match_string(MATCH_LIST *, const char *, const char *);
extern int match_hostname(MATCH_LIST *, const char *, const char *);
extern int match_hostaddr(MATCH_LIST *, const char *, const char *);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
