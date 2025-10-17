/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#include <stdarg.h>
#include <sys/malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syslog.h>

#include "LogMessage.h"
#include "webdavd.h"
#include "webdav_utils.h"

/*	-------------------------------------------------------------------------
        Debug printing defines
	------------------------------------------------------------------------- */
u_int32_t gPrintLevel = kNone | kSysLog | kAll;


void LogMessage(u_int32_t level, char *format, ...)
{
    va_list		arglist;
    int8_t		*buffer = NULL;
    int8_t		*bufPtr;

#if !DEBUG
	if (!(level & kSysLog)) {
		/* if we are not debugging, then only allow the syslog entries through */
		return;
	}
#endif
    if (level & gPrintLevel) {
		buffer = malloc (300);
		if (buffer == NULL)
			return;
		bzero (buffer, 300);
		
		bufPtr = buffer;

        va_start(arglist, format);
        vsnprintf ( (char*)buffer, 300, format, arglist);

        if (level & kSysLog) {
            syslog(LOG_DEBUG, "webdavfs: %s", buffer);
            level &= ~kSysLog;
        }

        if (level & gPrintLevel)
            syslog(LOG_DEBUG, "webdavfs: %s", buffer);
            //fprintf (stderr, (char*) "webdavfs: %s", buffer);
            
        va_end(arglist);
    }

    if (buffer != NULL) {
    	free(buffer);
        buffer = NULL;
    }
}

void logDebugCFString(const char *msg, CFStringRef str)
{
	char *cstr = NULL;
	
	if (str != NULL)
		cstr = createUTF8CStringFromCFString(str);
	
	if (msg == NULL) {
		if (cstr == NULL)
			return;	// nothing to do
		syslog(LOG_DEBUG, "%s\n", cstr);
	}
	else {
		if (cstr != NULL)
			syslog(LOG_DEBUG, "%s %s\n", msg, cstr);
		else
			syslog(LOG_DEBUG, "%s\n", msg);
	}

	if (cstr != NULL)
		free(cstr);
}

void logDebugCFURL(const char *msg, CFURLRef url)
{
	CFStringRef str = NULL;
	
	if (url != NULL) {
		str = CFURLGetString(url);
		CFRetain(str);
	}
	
	logDebugCFString(msg, str);
	
	if (str != NULL)
		CFRelease(str);
}

