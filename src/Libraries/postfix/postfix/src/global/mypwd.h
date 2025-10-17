/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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

#ifndef _MYPWNAM_H_INCLUDED_
#define _MYPWNAM_H_INCLUDED_

/*++
/* NAME
/*	mypwnam 3h
/* SUMMARY
/*	caching getpwnam_r()/getpwuid_r()
/* SYNOPSIS
/*	#include <mypwd.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
struct mypasswd {
    int     refcount;
    char   *pw_name;
    char   *pw_passwd;
    uid_t   pw_uid;
    gid_t   pw_gid;
    char   *pw_gecos;
    char   *pw_dir;
    char   *pw_shell;
};

extern int mypwnam_err(const char *, struct mypasswd **);
extern int mypwuid_err(uid_t, struct mypasswd **);
extern struct mypasswd *mypwnam(const char *);
extern struct mypasswd *mypwuid(uid_t);
extern void mypwfree(struct mypasswd *);

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
