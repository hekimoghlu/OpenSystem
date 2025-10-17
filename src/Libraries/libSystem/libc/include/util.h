/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
/*	$NetBSD: util.h,v 1.10 1997/12/01 02:25:46 lukem Exp $	*/

/*-
 * Copyright (c) 1995
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <sys/cdefs.h>
#include <sys/ttycom.h>
#include <sys/types.h>
#include <stdio.h>
#include <pwd.h>
#include <termios.h>
#include <Availability.h>

#define	PIDLOCK_NONBLOCK	1
#define PIDLOCK_USEHOSTNAME	2

/*
 * fparseln() specific operation flags.
 */
#define	FPARSELN_UNESCESC	0x01
#define	FPARSELN_UNESCCONT	0x02
#define	FPARSELN_UNESCCOMM	0x04
#define	FPARSELN_UNESCREST	0x08
#define	FPARSELN_UNESCALL	0x0f

/*
 * opendev() specific operation flags.
 */
#define OPENDEV_PART	0x01		/* Try to open the raw partition. */
#define OPENDEV_BLCK	0x04		/* Open block, not character device. */

__BEGIN_DECLS
#ifdef UNIFDEF_LEGACY_UTMP_APIS
struct utmp; /* forward reference to /usr/include/utmp.h */
void	login(struct utmp *)		__OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_5,__IPHONE_NA,__IPHONE_NA);
#endif /* UNIFDEF_LEGACY_UTMP_APIS */
int	login_tty(int);
#ifdef UNIFDEF_LEGACY_UTMP_APIS
int	logout(const char *)		__OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_5,__IPHONE_NA,__IPHONE_NA);
#endif /* UNIFDEF_LEGACY_UTMP_APIS */
void	logwtmp(const char *, const char *, const char *) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_5,__IPHONE_2_0,__IPHONE_2_0);
int	opendev(char *, int, int, char **);
int	openpty(int *, int *, char *, struct termios *,
		     struct winsize *);
char   *fparseln(FILE *, size_t *, size_t *, const char[3], int);
pid_t	forkpty(int *, char *, struct termios *, struct winsize *);
int	pidlock(const char *, int, pid_t *, const char *);
int	ttylock(const char *, int, pid_t *);
int	ttyunlock(const char *);
int	ttyaction(char *tty, char *act, char *user);
struct iovec;
char   *ttymsg(struct iovec *, int, const char *, int);
__END_DECLS
#ifdef UNIFDEF_LEGACY_UTMP_APIS

/* Include utmp.h last to avoid deprecation warning above */
#include <utmp.h>
#endif /* UNIFDEF_LEGACY_UTMP_APIS */

#endif /* !_UTIL_H_ */
