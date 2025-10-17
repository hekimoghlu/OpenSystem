/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#include "config.h"
#include "LibWebRTCLogSink.h"

#include <wtf/TZoneMallocInlines.h>

#if USE(LIBWEBRTC)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCLogSink);

LibWebRTCLogSink::LibWebRTCLogSink(LogCallback&& callback)
    : m_callback(WTFMove(callback))
{
}

LibWebRTCLogSink::~LibWebRTCLogSink()
{
    ASSERT(!m_loggingLevel);
}

void LibWebRTCLogSink::logMessage(const std::string& message, rtc::LoggingSeverity severity)
{
    m_callback(severity, message);
}

void LibWebRTCLogSink::start(rtc::LoggingSeverity level)
{
    if (level == rtc::LoggingSeverity::LS_NONE) {
        stop();
        return;
    }

    if (m_loggingLevel) {
        if (*m_loggingLevel == level)
            return;
        rtc::LogMessage::RemoveLogToStream(this);
    }

    m_loggingLevel = level;
    rtc::LogMessage::AddLogToStream(this, level);
}

void LibWebRTCLogSink::stop()
{
    if (!m_loggingLevel)
        return;

    m_loggingLevel = { };
    rtc::LogMessage::RemoveLogToStream(this);
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
