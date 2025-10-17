/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
//
// logging - message log support
//
#ifdef __MWERKS__
# define _CPP_LOGGING
#endif

#include <security_utilities/logging.h>
#include <security_utilities/globalizer.h>
#include <cstdarg>
#include <syslog.h>

namespace Security
{

namespace Syslog
{

//
// Open and initialize logging
//
void open(const char *ident, int facility, int options /*= 0*/)
{
	::openlog(ident, options, facility);
}


//
// General output method
//

static void output(int priority, const char *format, va_list args) __attribute__((format(printf, 2, 0)));


static void output(int priority, const char *format, va_list args)
{
	::vsyslog(priority, format, args);
}


//
// Priority-specific wrappers
//
void syslog(int priority, const char *format, ...)
{ va_list args; va_start(args, format); output(priority, format, args); va_end(args); }

void emergency(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_EMERG, format, args); va_end(args); }
void alert(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_ALERT, format, args); va_end(args); }
void critical(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_CRIT, format, args); va_end(args); }
void error(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_ERR, format, args); va_end(args); }
void warning(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_WARNING, format, args); va_end(args); }
void notice(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_NOTICE, format, args); va_end(args); }
void info(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_INFO, format, args); va_end(args); }
void debug(const char *format, ...)
{ va_list args; va_start(args, format); output(LOG_DEBUG, format, args); va_end(args); }


//
// Enable mask operation
//
int mask()
{
	int mask;
	::setlogmask(mask = ::setlogmask(0));
	return mask;
}
	
void upto(int priority)
{
	::setlogmask(LOG_UPTO(priority));
}

void enable(int priority)
{
	::setlogmask(::setlogmask(0) | LOG_MASK(priority));
}

void disable(int priority)
{
	::setlogmask(::setlogmask(0) & ~LOG_MASK(priority));
}

} // end namespace Syslog

} // end namespace Security
