/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#include "BPlatform.h"
#include "TZoneLog.h"

#if BUSE(TZONE)

#include "BAssert.h"
#include <mutex.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

namespace bmalloc { namespace api {

TZoneLog* TZoneLog::theTZoneLog = nullptr;

void TZoneLog::init()
{
    auto logEnv = getenv("TZONE_LOGGING");

#if BUSE(OS_LOG)
    // Enable OS Logging by default
    if (!logEnv)
        logEnv = const_cast<char*>("oslog");
#endif

    if (logEnv) {
        if (!strcasecmp(logEnv, "stderr"))
            m_logDest = LogDestination::Stderr;
#if BUSE(OS_LOG)
        else if (!strcasecmp(logEnv, "oslog")) {
            m_logDest = LogDestination::OSLog;
            m_osLog = os_log_create("com.apple.WebKit", "TZone");
        }
#endif
    }
}

extern TZoneLog& TZoneLog::singleton()
{
    if (!theTZoneLog)
        ensureSingleton();
    BASSERT(theTZoneLog);
    return *theTZoneLog;
}

BATTRIBUTE_PRINTF(2, 3)
extern void TZoneLog::log(const char* format, ...)
{
    if (LogDestination::Off)
        return;

    va_list argList;
    va_start(argList, format);
    if (m_logDest == LogDestination::OSLog)
        osLogWithLineBuffer(format, argList);
    else if (m_logDest == LogDestination::Stderr)
        vfprintf(stderr, format, argList);
    va_end(argList);
}

#if BUSE(OS_LOG)
BATTRIBUTE_PRINTF(2, 0)
void TZoneLog::osLogWithLineBuffer(const char* format, va_list list)
{
    if (!format)
        return;

    auto len = strlen(format);
    if (!len)
        return;

    char* nextBufferPtr = m_buffer + m_bufferCursor;
    bool endsWithNewline = format[strlen(format) - 1] == '\n';

    auto newCursor = vsnprintf(nextBufferPtr, s_osLogBufferSize - m_bufferCursor, format, list) + m_bufferCursor;

    if (endsWithNewline || newCursor >= s_osLogBufferSize) {
        // Dump current buffer
        os_log_debug(osLog(), "%s\n", m_buffer);
        m_bufferCursor = 0;
        return;
    }

    m_bufferCursor = newCursor;
}
#endif

void TZoneLog::ensureSingleton()
{
    static std::once_flag onceFlag;
    std::call_once(
        onceFlag,
        [] {
            theTZoneLog = new TZoneLog();
            theTZoneLog->init();
        }
    );
};

} } // namespace bmalloc::api

#endif // BUSE(TZONE)

