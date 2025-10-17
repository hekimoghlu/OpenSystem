/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
/* config.h - configurables for Vixie Cron
 *
 * $FreeBSD: src/usr.sbin/cron/cron/config.h,v 1.8 1999/08/28 01:15:49 peter Exp $
 */

#if !defined(_PATH_SENDMAIL)
# define _PATH_SENDMAIL "/usr/lib/sendmail"
#endif /*SENDMAIL*/

/*
 * these are site-dependent
 */

#ifndef DEBUGGING
#define DEBUGGING 1	/* 1 or 0 -- do you want debugging code built in? */
#endif

			/*
			 * choose one of these MAILCMD commands.  I use
			 * /bin/mail for speed; it makes biff bark but doesn't
			 * do aliasing.  /usr/lib/sendmail does aliasing but is
			 * a hog for short messages.  aliasing is not needed
			 * if you make use of the MAILTO= feature in crontabs.
			 * (hint: MAILTO= was added for this reason).
			 */

#define MAILCMD _PATH_SENDMAIL					/*-*/
#define MAILARGS "%s -FCronDaemon -odi -oem -oi -t"             /*-*/
			/* -Fx	 = set full-name of sender
			 * -odi	 = Option Deliverymode Interactive
			 * -oem	 = Option Errors Mailedtosender
			 * -oi   = Option dot message terminator
			 * -t    = read recipients from header of message
			 */

/* #define MAILCMD "/bin/mail" */		/*-*/
/* #define MAILARGS "%s -d  %s" */		/*-*/
			/* -d = undocumented but common flag: deliver locally?
			 */

/* #define MAILCMD "/usr/mmdf/bin/submit" */	/*-*/
/* #define MAILARGS "%s -mlrxto %s" */		/*-*/

/* #define MAIL_DATE */				/*-*/
			/* should we include an ersatz Date: header in
			 * generated mail?  if you are using sendmail
			 * for MAILCMD, it is better to let sendmail
			 * generate the Date: header.
			 */

			/* if ALLOW_FILE and DENY_FILE are not defined or are
			 * defined but neither exists, should crontab(1) be
			 * usable only by root?
			 */
/* #define ALLOW_ONLY_ROOT */			/*-*/

			/* if you want to use syslog(3) instead of appending
			 * to CRONDIR/LOG_FILE (/var/cron/log, e.g.), define
			 * SYSLOG here.  Note that quite a bit of logging
			 * info is written, and that you probably don't want
			 * to use this on 4.2bsd since everything goes in
			 * /usr/spool/mqueue/syslog.  On 4.[34]bsd you can
			 * tell /etc/syslog.conf to send cron's logging to
			 * a separate file.
			 *
			 * Note that if this and LOG_FILE in "pathnames.h"
			 * are both defined, then logging will go to both
			 * places.
			 */
#define SYSLOG	 			/*-*/
