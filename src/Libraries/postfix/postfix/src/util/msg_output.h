/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#ifndef _MSG_OUTPUT_FN_
#define _MSG_OUTPUT_FN_

/*++
/* NAME
/*	msg_output 3h
/* SUMMARY
/*	diagnostics output management
/* SYNOPSIS
/*	#include <msg_output.h>
/* DESCRIPTION

 /*
  * System library.
  */
#include <stdarg.h>

 /*
  * External interface. Severity levels are documented to be monotonically
  * increasing from 0 up to MSG_LAST.
  */
typedef void (*MSG_OUTPUT_FN) (int, const char *);
extern void msg_output(MSG_OUTPUT_FN);
extern void PRINTFLIKE(2, 3) msg_printf(int, const char *,...);
extern void msg_vprintf(int, const char *, va_list);
extern void msg_text(int, const char *);

#define MSG_INFO	0		/* informative */
#define	MSG_WARN	1		/* warning (non-fatal) */
#define MSG_ERROR	2		/* error (fatal) */
#define MSG_FATAL	3		/* software error (fatal) */
#define MSG_PANIC	4		/* software error (fatal) */

#define MSG_LAST	4		/* highest-numbered severity level */

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
