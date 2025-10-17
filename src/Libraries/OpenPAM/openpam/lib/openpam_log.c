/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>

#include <security/pam_appl.h>

#include "openpam_impl.h"

#ifdef OPENPAM_DEBUG
int _openpam_debug = 1;
#else
int _openpam_debug = 0;
#endif

#if !defined(openpam_log)

/*
 * OpenPAM extension
 *
 * Log a message through syslog
 */

void
openpam_log(int level, const char *fmt, ...)
{
	va_list ap;
	int priority;

	switch (level) {
	case PAM_LOG_LIBDEBUG:
		if (!_openpam_debug)
			return;
	case PAM_LOG_DEBUG:
		priority = LOG_DEBUG;
		break;
	case PAM_LOG_VERBOSE:
		priority = LOG_INFO;
		break;
	case PAM_LOG_NOTICE:
		priority = LOG_NOTICE;
		break;
	case PAM_LOG_ERROR:
	default:
		priority = LOG_ERR;
		break;
	}
	va_start(ap, fmt);
	vsyslog(LOG_AUTHPRIV|priority, fmt, ap);
	va_end(ap);
}

#else

void
_openpam_log(int level, const char *func, const char *fmt, ...)
{
	va_list ap;
	char *format;
	int priority;

	switch (level) {
	case PAM_LOG_LIBDEBUG:
		if (!_openpam_debug)
			return;
	case PAM_LOG_DEBUG:
		priority = LOG_DEBUG;
		break;
	case PAM_LOG_VERBOSE:
		priority = LOG_INFO;
		break;
	case PAM_LOG_NOTICE:
		priority = LOG_NOTICE;
		break;
	case PAM_LOG_ERROR:
	default:
		priority = LOG_ERR;
		break;
	}
	va_start(ap, fmt);
	if (asprintf(&format, "in %s(): %s", func, fmt) > 0) {
		vsyslog(LOG_AUTHPRIV|priority, format, ap);
		FREE(format);
	} else {
		vsyslog(LOG_AUTHPRIV|priority, fmt, ap);
	}
	va_end(ap);
}

#endif

/**
 * The =openpam_log function logs messages using =syslog.
 * It is primarily intended for internal use by the library and modules.
 *
 * The =level argument indicates the importance of the message.
 * The following levels are defined:
 *
 *	=PAM_LOG_DEBUG:
 *		Debugging messages.
 *		These messages are logged with a =syslog priority of
 *		=LOG_DEBUG.
 *	=PAM_LOG_VERBOSE:
 *		Information about the progress of the authentication
 *		process, or other non-essential messages.
 *		These messages are logged with a =syslog priority of
 *		=LOG_INFO.
 *	=PAM_LOG_NOTICE:
 *		Messages relating to non-fatal errors.
 *		These messages are logged with a =syslog priority of
 *		=LOG_NOTICE.
 *	=PAM_LOG_ERROR:
 *		Messages relating to serious errors.
 *		These messages are logged with a =syslog priority of
 *		=LOG_ERR.
 *
 * The remaining arguments are a =printf format string and the
 * corresponding arguments.
 */
