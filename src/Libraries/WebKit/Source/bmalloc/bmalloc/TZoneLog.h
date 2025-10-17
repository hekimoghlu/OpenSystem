/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#pragma once

#if BUSE(TZONE)

#include "BExport.h"
#include "BInline.h"
#include "BPlatform.h"

#include <os/log.h>
#include <stdarg.h>

namespace bmalloc { namespace api {

class TZoneLog {
public:
    enum LogDestination {
        Off,
        Stderr,
#if BUSE(OS_LOG)
        OSLog
#endif
    };

protected:
    TZoneLog()
        : m_logDest(LogDestination::Off)
    { }

public:
    TZoneLog(TZoneLog &other) = delete;
    void operator=(const TZoneLog &) = delete;

    BEXPORT static TZoneLog& singleton();
    BEXPORT void log(const char* format, ...) BATTRIBUTE_PRINTF(2, 3);

private:
    static void ensureSingleton();
    void init();
    static TZoneLog* theTZoneLog;

    BINLINE bool useStdErr() { return m_logDest == LogDestination::Stderr; }
#if BUSE(OS_LOG)
    BINLINE bool useOSLog() { return m_logDest == LogDestination::OSLog; }
    BINLINE os_log_t osLog() const { return m_osLog; }
    void osLogWithLineBuffer(const char* format, va_list) BATTRIBUTE_PRINTF(2, 0);
#endif

    LogDestination m_logDest;

#if BUSE(OS_LOG)
    os_log_t m_osLog;
    static constexpr unsigned s_osLogBufferSize = 121;
    char m_buffer[s_osLogBufferSize];
    unsigned m_bufferCursor { 0 };
#endif
};

} } // namespace bmalloc::api

#define TZONE_LOG_DEBUG(...) TZoneLog::singleton().log(__VA_ARGS__)

#else // not BUSE(TZONE)
#define TZONE_LOG_DEBUG(...)
#endif // BUSE(TZONE)
