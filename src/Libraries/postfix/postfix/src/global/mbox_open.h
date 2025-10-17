/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

#ifndef _MBOX_OPEN_H_INCLUDED_
#define _MBOX_OPEN_H_INCLUDED_

/*++
/* NAME
/*	mbox_open 3h
/* SUMMARY
/*	mailbox access
/* SYNOPSIS
/*	#include <mbox_open.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>
#include <vstring.h>
#include <safe_open.h>

 /*
  * Global library.
  */
#include <dsn_buf.h>

 /*
  * External interface.
  */
typedef struct {
    char   *path;			/* saved path, for dot_unlock */
    VSTREAM *fp;			/* open stream or null */
    int     locked;			/* what locks were set */
} MBOX;
extern MBOX *mbox_open(const char *, int, mode_t, struct stat *, uid_t, gid_t,
		               int, const char *, DSN_BUF *);
extern void mbox_release(MBOX *);
extern const char *mbox_dsn(int, const char *);

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
