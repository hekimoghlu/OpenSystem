/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "LogStream.h"

#if ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)

#include <wtf/OSObjectPtr.h>

#if HAVE(OS_SIGNPOST)
#include <wtf/SystemTracing.h>
#endif

#include "LogStreamMessages.h"
#include "Logging.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamServerConnection.h"

#define MESSAGE_CHECK(assertion, connection) MESSAGE_CHECK_BASE(assertion, connection)

namespace WebKit {

LogStream::LogStream(int32_t pid)
    : m_pid(pid)
{
}

LogStream::~LogStream()
{
}

void LogStream::stopListeningForIPC()
{
    assertIsMainRunLoop();
    if (RefPtr logStreamConnection = m_logStreamConnection)
        logStreamConnection->stopReceivingMessages(Messages::LogStream::messageReceiverName(), m_logStreamIdentifier->toUInt64());
}

void LogStream::logOnBehalfOfWebContent(std::span<const uint8_t> logSubsystem, std::span<const uint8_t> logCategory, std::span<const uint8_t> nullTerminatedLogString, uint8_t logType)
{
    ASSERT(!isMainRunLoop());

    auto isNullTerminated = [](std::span<const uint8_t> view) {
        return view.data() && !view.empty() && view.back() == '\0';
    };

    bool isValidLogType = logType == OS_LOG_TYPE_DEFAULT || logType == OS_LOG_TYPE_INFO || logType == OS_LOG_TYPE_DEBUG || logType == OS_LOG_TYPE_ERROR || logType == OS_LOG_TYPE_FAULT;
    MESSAGE_CHECK(isNullTerminated(nullTerminatedLogString) && isValidLogType, m_logStreamConnection->connection());

    // os_log_hook on sender side sends a null category and subsystem when logging to OS_LOG_DEFAULT.
    auto osLog = OSObjectPtr<os_log_t>();
    if (isNullTerminated(logSubsystem) && isNullTerminated(logCategory)) {
        auto subsystem = byteCast<char>(logSubsystem.data());
        auto category = byteCast<char>(logCategory.data());
        osLog = adoptOSObject(os_log_create(subsystem, category));
    }

    auto osLogPointer = osLog.get() ? osLog.get() : OS_LOG_DEFAULT;

#if HAVE(OS_SIGNPOST)
    if (WTFSignpostHandleIndirectLog(osLogPointer, m_pid, byteCast<char>(nullTerminatedLogString)))
        return;
#endif

    // Use '%{public}s' in the format string for the preprocessed string from the WebContent process.
    // This should not reveal any redacted information in the string, since it has already been composed in the WebContent process.
    os_log_with_type(osLogPointer, static_cast<os_log_type_t>(logType), "WP[%d] %{public}s", m_pid, byteCast<char>(nullTerminatedLogString).data());
}

void LogStream::setup(IPC::StreamServerConnectionHandle&& serverConnection, LogStreamIdentifier logStreamIdentifier, CompletionHandler<void(IPC::Semaphore& streamWakeUpSemaphore, IPC::Semaphore& streamClientWaitSemaphore)>&& completionHandler)
{
    m_logStreamIdentifier = logStreamIdentifier;
    m_logStreamConnection = IPC::StreamServerConnection::tryCreate(WTFMove(serverConnection), { });

    static NeverDestroyed<Ref<IPC::StreamConnectionWorkQueue>> logQueue = IPC::StreamConnectionWorkQueue::create("Log work queue"_s);

    if (RefPtr logStreamConnection = m_logStreamConnection) {
        logStreamConnection->open(logQueue.get());
        logStreamConnection->startReceivingMessages(*this, Messages::LogStream::messageReceiverName(), m_logStreamIdentifier->toUInt64());
        completionHandler(logQueue.get()->wakeUpSemaphore(), logStreamConnection->clientWaitSemaphore());
    }
}

#if __has_include("LogMessagesImplementations.h")
#include "LogMessagesImplementations.h"
#endif

}

#endif
