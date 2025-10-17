/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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

#ifndef _SYS_EXITS_H_INCLUDED_
#define _SYS_EXITS_H_INCLUDED_

/*++
/* NAME
/*	sys_exits 3h
/* SUMMARY
/*	sendmail-compatible exit status handling
/* SYNOPSIS
/*	#include <sys_exits.h>
/* DESCRIPTION
/* .nf

 /*
  * External interface.
  */
typedef struct {
    const int status;			/* exit status code */
    const char *dsn;			/* DSN detail */
    const char *text;			/* descriptive text */
} SYS_EXITS_DETAIL;

extern const char *sys_exits_strerror(int);
extern const SYS_EXITS_DETAIL *sys_exits_detail(int);
extern int sys_exits_softerror(int);

#define SYS_EXITS_CODE(n) ((n) >= EX__BASE && (n) <= EX__MAX)

#define EX__BASE	64		/* base value for error messages */

#define EX_USAGE	64		/* command line usage error */
#define EX_DATAERR	65		/* data format error */
#define EX_NOINPUT	66		/* cannot open input */
#define EX_NOUSER	67		/* addressee unknown */
#define EX_NOHOST	68		/* host name unknown */
#define EX_UNAVAILABLE	69		/* service unavailable */
#define EX_SOFTWARE	70		/* internal software error */
#define EX_OSERR	71		/* system error (e.g., can't fork) */
#define EX_OSFILE	72		/* critical OS file missing */
#define EX_CANTCREAT	73		/* can't create (user) output file */
#define EX_IOERR	74		/* input/output error */
#define EX_TEMPFAIL	75		/* temporary failure */
#define EX_PROTOCOL	76		/* remote error in protocol */
#define EX_NOPERM	77		/* permission denied */
#define EX_CONFIG	78		/* configuration error */

#define EX__MAX	78			/* maximum listed value */

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
