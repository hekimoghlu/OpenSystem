/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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

#ifndef _SMTP_STREAM_H_INCLUDED_
#define _SMTP_STREAM_H_INCLUDED_

/*++
/* NAME
/*	smtp_stream 3h
/* SUMMARY
/*	smtp stream I/O support
/* SYNOPSIS
/*	#include <smtp_stream.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <stdarg.h>
#include <setjmp.h>

 /*
  * Utility library.
  */
#include <vstring.h>
#include <vstream.h>

 /*
  * External interface. The following codes are meant for use in longjmp(),
  * so they must all be non-zero.
  */
#define SMTP_ERR_EOF	1		/* unexpected client disconnect */
#define SMTP_ERR_TIME	2		/* time out */
#define SMTP_ERR_QUIET	3		/* silent cleanup (application) */
#define SMTP_ERR_NONE	4		/* non-error case */
#define SMTP_ERR_DATA	5		/* application data error */

extern void smtp_stream_setup(VSTREAM *, int, int);
extern void PRINTFLIKE(2, 3) smtp_printf(VSTREAM *, const char *,...);
extern void smtp_flush(VSTREAM *);
extern int smtp_fgetc(VSTREAM *);
extern int smtp_get(VSTRING *, VSTREAM *, ssize_t, int);
extern void smtp_fputs(const char *, ssize_t len, VSTREAM *);
extern void smtp_fwrite(const char *, ssize_t len, VSTREAM *);
extern void smtp_fputc(int, VSTREAM *);

extern void smtp_vprintf(VSTREAM *, const char *, va_list);

#define smtp_timeout_setup(stream, timeout) \
	smtp_stream_setup((stream), (timeout), 0)

#define SMTP_GET_FLAG_NONE	0
#define SMTP_GET_FLAG_SKIP	(1<<0)	/* skip over excess input */

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
