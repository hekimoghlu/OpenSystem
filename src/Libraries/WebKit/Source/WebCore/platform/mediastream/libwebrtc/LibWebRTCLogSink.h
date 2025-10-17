/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/rtc_base/logging.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {
class LibWebRTCLogSink;
}

namespace WebCore {

class LibWebRTCLogSink final : rtc::LogSink {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCLogSink);
public:
    using LogCallback = Function<void(rtc::LoggingSeverity, const std::string&)>;
    explicit LibWebRTCLogSink(LogCallback&&);

    ~LibWebRTCLogSink();

    void start(rtc::LoggingSeverity = rtc::LoggingSeverity::LS_VERBOSE);
    void stop();

private:
    void OnLogMessage(const std::string& message, rtc::LoggingSeverity severity) final { logMessage(message, severity); }
    void OnLogMessage(const std::string& message) final { logMessage(message, rtc::LoggingSeverity::LS_INFO); }

    void logMessage(const std::string&, rtc::LoggingSeverity);

    LogCallback m_callback;
    std::optional<rtc::LoggingSeverity> m_loggingLevel;
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
