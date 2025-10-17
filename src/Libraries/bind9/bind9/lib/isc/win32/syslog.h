/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
/* $Id: syslog.h,v 1.7 2007/06/19 23:47:19 tbox Exp $ */

#ifndef _SYSLOG_H
#define _SYSLOG_H

#include <stdio.h>

/* Constant definitions for openlog() */
#define LOG_PID		1
#define LOG_CONS	2
/* NT event log does not support facility level */
#define LOG_KERN	0
#define LOG_USER	0
#define LOG_MAIL	0
#define LOG_DAEMON	0
#define LOG_AUTH	0
#define LOG_SYSLOG	0
#define LOG_LPR		0
#define LOG_LOCAL0	0
#define LOG_LOCAL1	0
#define LOG_LOCAL2	0
#define LOG_LOCAL3	0
#define LOG_LOCAL4	0
#define LOG_LOCAL5	0
#define LOG_LOCAL6	0
#define LOG_LOCAL7	0

#define LOG_EMERG       0       /* system is unusable */
#define LOG_ALERT       1       /* action must be taken immediately */
#define LOG_CRIT        2       /* critical conditions */
#define LOG_ERR         3       /* error conditions */
#define LOG_WARNING     4       /* warning conditions */
#define LOG_NOTICE      5       /* normal but signification condition */
#define LOG_INFO        6       /* informational */
#define LOG_DEBUG       7       /* debug-level messages */

void
syslog(int level, const char *fmt, ...);

void
openlog(const char *, int, ...);

void
closelog(void);

void
ModifyLogLevel(int level);

void
InitNTLogging(FILE *, int);

void
NTReportError(const char *, const char *);
/*
 * Include the event codes required for logging.
 */
#include <isc/bindevt.h>

#endif
