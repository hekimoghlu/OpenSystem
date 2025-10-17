/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#include "WorkerMessagePortChannelProvider.h"

#include "MessagePort.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerThread.h"
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerMessagePortChannelProvider);

WorkerMessagePortChannelProvider::WorkerMessagePortChannelProvider(WorkerOrWorkletGlobalScope& scope)
    : m_scope(scope)
{
}

WorkerMessagePortChannelProvider::~WorkerMessagePortChannelProvider()
{
    while (!m_takeAllMessagesCallbacks.isEmpty()) {
        auto first = m_takeAllMessagesCallbacks.takeFirst();
        first({ }, [] { });
    }
}

void WorkerMessagePortChannelProvider::createNewMessagePortChannel(const MessagePortIdentifier& local, const MessagePortIdentifier& remote)
{
    callOnMainThread([local, remote] {
        MessagePortChannelProvider::singleton().createNewMessagePortChannel(local, remote);
    });
}

void WorkerMessagePortChannelProvider::entangleLocalPortInThisProcessToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote)
{
    callOnMainThread([local, remote] {
        MessagePortChannelProvider::singleton().entangleLocalPortInThisProcessToRemote(local, remote);
    });
}

void WorkerMessagePortChannelProvider::messagePortDisentangled(const MessagePortIdentifier& local)
{
    callOnMainThread([local] {
        MessagePortChannelProvider::singleton().messagePortDisentangled(local);
    });
}

void WorkerMessagePortChannelProvider::messagePortClosed(const MessagePortIdentifier&)
{
    ASSERT_NOT_REACHED();
}

void WorkerMessagePortChannelProvider::postMessageToRemote(MessageWithMessagePorts&& message, const MessagePortIdentifier& remoteTarget)
{
    callOnMainThread([message = WTFMove(message), remoteTarget]() mutable {
        MessagePortChannelProvider::singleton().postMessageToRemote(WTFMove(message), remoteTarget);
    });
}

class MainThreadCompletionHandler {
public:
    explicit MainThreadCompletionHandler(CompletionHandler<void()>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    {
    }
    MainThreadCompletionHandler(MainThreadCompletionHandler&&) = default;
    MainThreadCompletionHandler& operator=(MainThreadCompletionHandler&&) = default;

    ~MainThreadCompletionHandler()
    {
        if (m_completionHandler)
            complete();
    }

    void complete()
    {
        callOnMainThread(WTFMove(m_completionHandler));
    }

private:
    CompletionHandler<void()> m_completionHandler;
};

void WorkerMessagePortChannelProvider::takeAllMessagesForPort(const MessagePortIdentifier& identifier, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&& callback)
{
    uint64_t callbackIdentifier = ++m_lastCallbackIdentifier;
    m_takeAllMessagesCallbacks.add(callbackIdentifier, WTFMove(callback));

    callOnMainThread([weakThis = WeakPtr { *this }, workerThread = RefPtr { m_scope->workerOrWorkletThread() }, callbackIdentifier, identifier]() mutable {
        MessagePortChannelProvider::singleton().takeAllMessagesForPort(identifier, [weakThis = WTFMove(weakThis), workerThread = WTFMove(workerThread), callbackIdentifier](Vector<MessageWithMessagePorts>&& messages, Function<void()>&& completionHandler) mutable {
            workerThread->runLoop().postTaskForMode([weakThis = WTFMove(weakThis), callbackIdentifier, messages = WTFMove(messages), completionHandler = MainThreadCompletionHandler(WTFMove(completionHandler))](auto&) mutable {
                CheckedPtr checkedThis = weakThis.get();
                if (!checkedThis)
                    return;
                checkedThis->m_takeAllMessagesCallbacks.take(callbackIdentifier)(WTFMove(messages), [completionHandler = WTFMove(completionHandler)]() mutable {
                    completionHandler.complete();
                });
            }, WorkerRunLoop::defaultMode());
        });
    });
}

} // namespace WebCore
