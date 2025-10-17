/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include "MessageNames.h"
#include <wtf/Forward.h>
#include <wtf/MachSendRight.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>


namespace Messages {
namespace TestWithStream {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithStream;
}

class SendString {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_SendString; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = true;
    static constexpr bool isStreamBatched = false;

    explicit SendString(const String& url)
        : m_arguments(url)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const String&> m_arguments;
};

class SendStringAsync {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_SendStringAsync; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = true;
    static constexpr bool isReplyStreamEncodable = true;
    static constexpr bool isStreamBatched = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithStream_SendStringAsyncReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<int64_t>;
    using Reply = CompletionHandler<void(int64_t)>;
    using Promise = WTF::NativePromise<int64_t, IPC::Error>;
    explicit SendStringAsync(const String& url)
        : m_arguments(url)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const String&> m_arguments;
};

class SendStringSync {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_SendStringSync; }
    static constexpr bool isSync = true;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = true;
    static constexpr bool isReplyStreamEncodable = true;
    static constexpr bool isStreamBatched = false;

    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<int64_t>;
    using Reply = CompletionHandler<void(int64_t)>;
    explicit SendStringSync(const String& url)
        : m_arguments(url)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const String&> m_arguments;
};

class CallWithIdentifier {
public:
    using Arguments = std::tuple<>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_CallWithIdentifier; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = true;
    static constexpr bool isReplyStreamEncodable = true;
    static constexpr bool isStreamBatched = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithStream_CallWithIdentifierReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<>;
    using Reply = CompletionHandler<void()>;
    using Promise = WTF::NativePromise<void, IPC::Error>;
    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<> m_arguments;
};

#if PLATFORM(COCOA)
class SendMachSendRight {
public:
    using Arguments = std::tuple<MachSendRight>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_SendMachSendRight; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = false;
    static constexpr bool isStreamBatched = false;

    explicit SendMachSendRight(MachSendRight&& a1)
        : m_arguments(WTFMove(a1))
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<MachSendRight&&> m_arguments;
};
#endif

#if PLATFORM(COCOA)
class ReceiveMachSendRight {
public:
    using Arguments = std::tuple<>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_ReceiveMachSendRight; }
    static constexpr bool isSync = true;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = true;
    static constexpr bool isReplyStreamEncodable = false;
    static constexpr bool isStreamBatched = false;

    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<MachSendRight>;
    using Reply = CompletionHandler<void(MachSendRight&&)>;
    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<> m_arguments;
};
#endif

#if PLATFORM(COCOA)
class SendAndReceiveMachSendRight {
public:
    using Arguments = std::tuple<MachSendRight>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStream_SendAndReceiveMachSendRight; }
    static constexpr bool isSync = true;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;
    static constexpr bool isStreamEncodable = false;
    static constexpr bool isReplyStreamEncodable = false;
    static constexpr bool isStreamBatched = false;

    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<MachSendRight>;
    using Reply = CompletionHandler<void(MachSendRight&&)>;
    explicit SendAndReceiveMachSendRight(MachSendRight&& a1)
        : m_arguments(WTFMove(a1))
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<MachSendRight&&> m_arguments;
};
#endif

} // namespace TestWithStream
} // namespace Messages
