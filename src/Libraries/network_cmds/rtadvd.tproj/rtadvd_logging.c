/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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
#include <assert.h>
#include <os/log_private.h>

#define kRtadvdLoggerID        "com.apple.rtadvd"
static os_log_t rtadvdLogger = NULL;                        /* Handle for Logger */

static boolean_t rtadvd_logger_create(void);

static boolean_t
rtadvd_logger_create(void)
{
    assert(rtadvdLogger == NULL);
    rtadvdLogger = os_log_create(kRtadvdLoggerID, "daemon");

    if (rtadvdLogger == NULL) {
        os_log_error(OS_LOG_DEFAULT, "Couldn't create os log object");
    }

    return (rtadvdLogger != NULL);
}

void
rtadvdLog(int level, const char *format, ...)
{
    va_list args;

    if (rtadvdLogger == NULL && !rtadvd_logger_create()) {
        return;
    }

    va_start(args, format);
    os_log_with_args(rtadvdLogger, level, format, args, __builtin_return_address(0));
    va_end(args);
    return;
}
