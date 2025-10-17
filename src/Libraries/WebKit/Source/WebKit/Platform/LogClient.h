/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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

#if ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)

#include "LogStream.h"
#include "LogStreamIdentifier.h"
#include "LogStreamMessages.h"
#include "StreamClientConnection.h"
#include <WebCore/LogClient.h>
#include <wtf/Lock.h>

#if __has_include("WebCoreLogDefinitions.h")
#include "WebCoreLogDefinitions.h"
#endif
#if __has_include("WebKitLogDefinitions.h")
#include "WebKitLogDefinitions.h"
#endif

namespace WebKit {

class LogClient final : public WebCore::LogClient {
    WTF_MAKE_FAST_ALLOCATED;
public:
    LogClient(IPC::StreamClientConnection&, const LogStreamIdentifier&);

    void log(std::span<const uint8_t> logChannel, std::span<const uint8_t> logCategory, std::span<const uint8_t> logString, os_log_type_t) final;

#if __has_include("WebKitLogClientDeclarations.h")
#include "WebKitLogClientDeclarations.h"
#endif

#if __has_include("WebCoreLogClientDeclarations.h")
#include "WebCoreLogClientDeclarations.h"
#endif

private:
    bool isWebKitLogClient() const final { return true; }

    const Ref<IPC::StreamClientConnection> m_logStreamConnection WTF_GUARDED_BY_LOCK(m_logStreamLock);
    LogStreamIdentifier m_logStreamIdentifier;
    Lock m_logStreamLock;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::LogClient)
static bool isType(const WebCore::LogClient& logClient) { return logClient.isWebKitLogClient(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)
