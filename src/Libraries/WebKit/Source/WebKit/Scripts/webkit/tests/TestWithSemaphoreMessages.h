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
#pragma once

#include "ArgumentCoders.h"
#include "Connection.h"
#include "IPCSemaphore.h"
#include "MessageNames.h"
#include <wtf/Forward.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/ThreadSafeRefCounted.h>


namespace Messages {
namespace TestWithSemaphore {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithSemaphore;
}

class SendSemaphore {
public:
    using Arguments = std::tuple<IPC::Semaphore>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSemaphore_SendSemaphore; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    explicit SendSemaphore(const IPC::Semaphore& s0)
        : m_arguments(s0)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const IPC::Semaphore&> m_arguments;
};

class ReceiveSemaphore {
public:
    using Arguments = std::tuple<>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSemaphore_ReceiveSemaphore; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithSemaphore_ReceiveSemaphoreReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<IPC::Semaphore>;
    using Reply = CompletionHandler<void(IPC::Semaphore&&)>;
    using Promise = WTF::NativePromise<IPC::Semaphore, IPC::Error>;
    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<> m_arguments;
};

} // namespace TestWithSemaphore
} // namespace Messages
