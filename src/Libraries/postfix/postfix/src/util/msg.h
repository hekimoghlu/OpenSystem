/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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

#ifndef _MSG_H_INCLUDED_
#define _MSG_H_INCLUDED_

/*++
/* NAME
/*	msg 3h
/* SUMMARY
/*	diagnostics interface
/* SYNOPSIS
/*	#include "msg.h"
/* DESCRIPTION
/*	.nf

/*
 * System library.
 */
#include <stdarg.h>
#include <time.h>

/*
 * External interface.
 */
typedef void (*MSG_CLEANUP_FN) (void);

extern int msg_verbose;

extern void PRINTFLIKE(1, 2) msg_info(const char *,...);
extern void PRINTFLIKE(1, 2) msg_warn(const char *,...);
extern void PRINTFLIKE(1, 2) msg_error(const char *,...);
extern NORETURN PRINTFLIKE(1, 2) msg_fatal(const char *,...);
extern NORETURN PRINTFLIKE(2, 3) msg_fatal_status(int, const char *,...);
extern NORETURN PRINTFLIKE(1, 2) msg_panic(const char *,...);

extern void vmsg_info(const char *, va_list);
extern void vmsg_warn(const char *, va_list);
extern void vmsg_error(const char *, va_list);
extern NORETURN vmsg_fatal(const char *, va_list);
extern NORETURN vmsg_fatal_status(int, const char *, va_list);
extern NORETURN vmsg_panic(const char *, va_list);

extern int msg_error_limit(int);
extern void msg_error_clear(void);
extern MSG_CLEANUP_FN msg_cleanup(MSG_CLEANUP_FN);

extern void PRINTFLIKE(4, 5) msg_rate_delay(time_t *, int,
	              void PRINTFPTRLIKE(1, 2) (*log_fn) (const char *,...),
					            const char *,...);

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
