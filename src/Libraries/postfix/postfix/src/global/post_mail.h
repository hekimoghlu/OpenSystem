/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

#ifndef _POST_MAIL_H_INCLUDED_
#define _POST_MAIL_H_INCLUDED_

/*++
/* NAME
/*	post_mail 3h
/* SUMMARY
/*	convenient mail posting interface
/* SYNOPSIS
/*	#include <post_mail.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>
#include <vstring.h>

 /*
  * Global library.
  */
#include <cleanup_user.h>
#include <mail_proto.h>
#include <smtputf8.h>
#include <int_filt.h>

 /*
  * External interface.
  */
typedef void (*POST_MAIL_NOTIFY) (VSTREAM *, void *);
extern VSTREAM *post_mail_fopen(const char *, const char *, int, int, int, VSTRING *);
extern VSTREAM *post_mail_fopen_nowait(const char *, const char *, int, int, int, VSTRING *);
extern void post_mail_fopen_async(const char *, const char *, int, int, int, VSTRING *, POST_MAIL_NOTIFY, void *);
extern int PRINTFLIKE(2, 3) post_mail_fprintf(VSTREAM *, const char *,...);
extern int post_mail_fputs(VSTREAM *, const char *);
extern int post_mail_buffer(VSTREAM *, const char *, int);
extern int post_mail_fclose(VSTREAM *);
typedef void (*POST_MAIL_FCLOSE_NOTIFY) (int, void *);
extern void post_mail_fclose_async(VSTREAM *, POST_MAIL_FCLOSE_NOTIFY, void *);

#define POST_MAIL_BUFFER(v, b) \
	post_mail_buffer((v), vstring_str(b), VSTRING_LEN(b))

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
