/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#include <config.h>
#include "dcethread-debug.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;

void
dcethread__default_log_callback (const char* file, unsigned int line,
	int level, const char* str, void* data ATTRIBUTE_UNUSED)
{
    const char* level_name = NULL;

    switch (level)
    {
    case DCETHREAD_DEBUG_ERROR:
        level_name = "ERROR";
        break;
    case DCETHREAD_DEBUG_WARNING:
        level_name = "WARNING";
        break;
    case DCETHREAD_DEBUG_INFO:
        level_name = "INFO";
        break;
    case DCETHREAD_DEBUG_VERBOSE:
        level_name = "VERBOSE";
        break;
    case DCETHREAD_DEBUG_TRACE:
        level_name = "TRACE";
        break;
    default:
        level_name = "UNKNOWN";
        break;
    }

    pthread_mutex_lock(&log_lock);
    fprintf(stderr, "dcethread-%s %s:%i: %s\n", level_name, file, line, str);
    if (level == DCETHREAD_DEBUG_ERROR)
        abort();
    pthread_mutex_unlock(&log_lock);
}

static void (*log_callback) (const char* file, unsigned int line, int level, const char* str, void* data) = NULL;
static void *log_callback_data = NULL;

void
dcethread__debug_set_callback(void (*cb) (const char*, unsigned int, int, const char*, void* data), void* data)
{
    log_callback = cb;
    log_callback_data = data;
}

#ifndef HAVE_VASPRINTF
static char *
my_vasprintf(const char* format, va_list args)
{
    char *smallBuffer;
    unsigned int bufsize;
    int requiredLength;
    int newRequiredLength;
    char* outputString = NULL;
    va_list args2;

    va_copy(args2, args);

    bufsize = 4;
    /* Use a small buffer in case libc does not like NULL */
    do
    {
        smallBuffer = malloc(bufsize);

	if (!smallBuffer)
	{
	    return NULL;
	}

        requiredLength = vsnprintf(smallBuffer, bufsize, format, args);
        if (requiredLength < 0)
        {
            bufsize *= 2;
        }
	free(smallBuffer);
    } while (requiredLength < 0);

    if (requiredLength >= (int)(0xFFFFFFFF - 1))
    {
        return NULL;
    }

    outputString = malloc(requiredLength + 2);

    if (!outputString)
    {
	return NULL;
    }

    newRequiredLength = vsnprintf(outputString, requiredLength + 1, format, args2);
    if (newRequiredLength < 0)
    {
	free(outputString);
	return NULL;
    }

    va_end(args2);

    return outputString;
}
#endif /* HAVE_VASPRINTF */

void
dcethread__debug_printf(const char* file, unsigned int line, int level, const char* fmt, ...)
{
    va_list ap;
    char* str = NULL;

    if (!log_callback)
	return;

    va_start(ap, fmt);

#if HAVE_VASPRINTF
    vasprintf(&str, fmt, ap);
#else
    str = my_vasprintf(fmt, ap);
#endif

    if (str)
    {
	log_callback(file, line, level, str, log_callback_data);
	free(str);
    }

    va_end(ap);
}
