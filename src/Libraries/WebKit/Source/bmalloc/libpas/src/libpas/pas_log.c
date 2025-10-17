/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_log.h"

#include <errno.h>
#include "pas_snprintf.h"
#include <unistd.h>

pthread_t pas_thread_that_is_crash_logging;

// Debug option to log to a file instead of stdout by default.
// This does not affect pas_fd_stream.
#define PAS_DEBUG_LOG_TO_SYSLOG 0

#if PAS_DEBUG_LOG_TO_SYSLOG
#include <sys/syslog.h>
#endif

void pas_vlog_fd(int fd, const char* format, va_list list)
{
    char buf[PAS_LOG_MAX_BYTES];
    ssize_t result;
    char* ptr;
    size_t bytes_left_to_write;
    pthread_t thread_that_is_crash_logging;

    thread_that_is_crash_logging = pas_thread_that_is_crash_logging;
    while (thread_that_is_crash_logging && thread_that_is_crash_logging != pthread_self()) {
        pas_compiler_fence();
        thread_that_is_crash_logging = pas_thread_that_is_crash_logging;
    }

    result = pas_vsnprintf(buf, PAS_LOG_MAX_BYTES, format, list);

    PAS_ASSERT(result >= 0);

    if ((size_t)result < PAS_LOG_MAX_BYTES)
        bytes_left_to_write = (size_t)result;
    else
        bytes_left_to_write = PAS_LOG_MAX_BYTES - 1;

    ptr = buf;

    while (bytes_left_to_write) {
        result = write(fd, ptr, bytes_left_to_write);
        if (result < 0) {
            PAS_ASSERT(errno == EINTR);
            continue;
        }

        PAS_ASSERT(result);

        ptr += result;
        bytes_left_to_write -= (size_t)result;
    }
}

void pas_log_fd(int fd, const char* format, ...)
{
    va_list arg_list;
    va_start(arg_list, format);
    pas_vlog_fd(fd, format, arg_list);
    va_end(arg_list);
}

void pas_vlog(const char* format, va_list list)
{
#if PAS_DEBUG_LOG_TO_SYSLOG
PAS_IGNORE_WARNINGS_BEGIN("format-nonliteral")
    syslog(LOG_WARNING, format, list);
PAS_IGNORE_WARNINGS_END
#else
    pas_vlog_fd(PAS_LOG_DEFAULT_FD, format, list);
#endif
}

void pas_log(const char* format, ...)
{
    va_list arg_list;
    va_start(arg_list, format);
    pas_vlog(format, arg_list);
    va_end(arg_list);
}

void pas_start_crash_logging(void)
{
    pas_thread_that_is_crash_logging = pthread_self();
    pas_fence();
}

#endif /* LIBPAS_ENABLED */
