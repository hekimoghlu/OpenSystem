/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#pragma prototyped
/*
 * posix syslog implementation definitions
 */

#ifndef _SYSLOGLIB_H
#define _SYSLOGLIB_H

#include <syslog.h>

#define log		_log_info_
#define sendlog		_log_send_

/*
 * NOTE: syslog() has a static initializer for Syslog_state_t log
 */

typedef struct
{
	int		facility;	/* openlog facility		*/
	int		fd;		/* log to this fd		*/
	int		flags;		/* openlog flags		*/
	unsigned int	mask;		/* setlogmask mask		*/
	int		attempt;	/* logfile attempt state	*/
	char		ident[64];	/* openlog ident		*/
	char		host[64];	/* openlog host name		*/
} Syslog_state_t;

extern Syslog_state_t	log;

extern void		sendlog(const char*);

#endif
