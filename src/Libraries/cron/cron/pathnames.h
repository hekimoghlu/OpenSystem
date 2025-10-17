/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
 * $FreeBSD: src/usr.sbin/cron/cron/pathnames.h,v 1.5 1999/08/28 01:15:50 peter Exp $
 */

#if (defined(BSD)) && (BSD >= 199103) || defined(__linux) || defined(AIX)
# include <paths.h>
#endif /*BSD*/

#ifndef CRONDIR
			/* CRONDIR is where crond(8) and crontab(1) both chdir
			 * to; SPOOL_DIR, ALLOW_FILE, DENY_FILE, and LOG_FILE
			 * are all relative to this directory.
			 */
#ifdef __APPLE__
#define CRONDIR		"/usr/lib/cron"
#else
#define CRONDIR		"/var/cron"
#endif
#endif

			/* SPOOLDIR is where the crontabs live.
			 * This directory will have its modtime updated
			 * whenever crontab(1) changes a crontab; this is
			 * the signal for crond(8) to look at each individual
			 * crontab file and reload those whose modtimes are
			 * newer than they were last time around (or which
			 * didn't exist last time around...)
			 */
#define SPOOL_DIR	"tabs"
#define TMP_SPOOL_DIR	"tmp"

			/* undefining these turns off their features.  note
			 * that ALLOW_FILE and DENY_FILE must both be defined
			 * in order to enable the allow/deny code.  If neither
			 * LOG_FILE or SYSLOG is defined, we don't log.  If
			 * both are defined, we log both ways.
			 */
#ifdef __APPLE__
#define	ALLOW_FILE	"cron.allow"		/*-*/
#define DENY_FILE	"cron.deny"		/*-*/
#else
#define	ALLOW_FILE	"allow"		/*-*/
#define DENY_FILE	"deny"		/*-*/
#endif
/*#define LOG_FILE        "log"*/           /*-*/

			/* where should the daemon stick its PID?
			 */
#ifdef _PATH_VARRUN
# define PIDDIR	_PATH_VARRUN
#else
# define PIDDIR "/etc/"
#endif
#define PIDFILE		"%scron.pid"

			/* 4.3BSD-style crontab */
#define SYSCRONTAB	"/etc/crontab"

			/* what editor to use if no EDITOR or VISUAL
			 * environment variable specified.
			 */
#if defined(_PATH_VI)
# define EDITOR _PATH_VI
#else
# define EDITOR "/usr/ucb/vi"
#endif

#ifndef _PATH_BSHELL
# define _PATH_BSHELL "/bin/sh"
#endif

#ifndef _PATH_DEFPATH
# define _PATH_DEFPATH "/usr/bin:/bin"
#endif
