/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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

#include "LogStreamIdentifier.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamMessageReceiver.h"

#include <wtf/RefPtr.h>

namespace IPC {
class StreamServerConnection;
struct StreamServerConnectionHandle;
}

namespace WebKit {

class LogStream : public IPC::StreamMessageReceiver {
public:
    static Ref<LogStream> create(int32_t pid) { return adoptRef(*new LogStream(pid)); }
    ~LogStream();

    void setup(IPC::StreamServerConnectionHandle&&, LogStreamIdentifier, CompletionHandler<void(IPC::Semaphore& streamWakeUpSemaphore, IPC::Semaphore& streamClientWaitSemaphore)>&&);
    void stopListeningForIPC();

private:
    LogStream(int32_t pid);

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    void logOnBehalfOfWebContent(std::span<const uint8_t> logChannel, std::span<const uint8_t> logCategory, std::span<const uint8_t> logString, uint8_t logType);

#if __has_include("LogMessagesDeclarations.h")
#include "LogMessagesDeclarations.h"
#endif

    RefPtr<IPC::StreamServerConnection> m_logStreamConnection;
    int32_t m_pid { 0 };
    Markable<LogStreamIdentifier> m_logStreamIdentifier;
};

} // namespace WebKit

#endif // ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)
