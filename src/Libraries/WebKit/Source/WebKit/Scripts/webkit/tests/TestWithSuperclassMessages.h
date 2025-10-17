/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include "TestClassName.h"
#include <optional>
#include <wtf/Forward.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
enum class TestTwoStateEnum : bool;
}

namespace Messages {
namespace TestWithSuperclass {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithSuperclass;
}

class LoadURL {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_LoadURL; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    explicit LoadURL(const String& url)
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

#if ENABLE(TEST_FEATURE)
class TestAsyncMessage {
public:
    using Arguments = std::tuple<WebKit::TestTwoStateEnum>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessage; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::MainThread;
    using ReplyArguments = std::tuple<uint64_t>;
    using Reply = CompletionHandler<void(uint64_t)>;
    using Promise = WTF::NativePromise<uint64_t, IPC::Error>;
    explicit TestAsyncMessage(WebKit::TestTwoStateEnum twoStateEnum)
        : m_arguments(twoStateEnum)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<WebKit::TestTwoStateEnum> m_arguments;
};
#endif

#if ENABLE(TEST_FEATURE)
class TestAsyncMessageWithNoArguments {
public:
    using Arguments = std::tuple<>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageWithNoArguments; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageWithNoArgumentsReply; }
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
#endif

#if ENABLE(TEST_FEATURE)
class TestAsyncMessageWithMultipleArguments {
public:
    using Arguments = std::tuple<>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageWithMultipleArguments; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageWithMultipleArgumentsReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<bool, uint64_t>;
    using Reply = CompletionHandler<void(bool, uint64_t)>;
    using Promise = WTF::NativePromise<std::tuple<bool, uint64_t>, IPC::Error>;
    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<> m_arguments;
};
#endif

#if ENABLE(TEST_FEATURE)
class TestAsyncMessageWithConnection {
public:
    using Arguments = std::tuple<int>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageWithConnection; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithSuperclass_TestAsyncMessageWithConnectionReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<bool>;
    using Reply = CompletionHandler<void(bool)>;
    using Promise = WTF::NativePromise<bool, IPC::Error>;
    explicit TestAsyncMessageWithConnection(const int& value)
        : m_arguments(value)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const int&> m_arguments;
};
#endif

class TestSyncMessage {
public:
    using Arguments = std::tuple<uint32_t>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_TestSyncMessage; }
    static constexpr bool isSync = true;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<uint8_t>;
    using Reply = CompletionHandler<void(uint8_t)>;
    explicit TestSyncMessage(uint32_t param)
        : m_arguments(param)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<uint32_t> m_arguments;
};

class TestSynchronousMessage {
public:
    using Arguments = std::tuple<bool>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithSuperclass_TestSynchronousMessage; }
    static constexpr bool isSync = true;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<std::optional<WebKit::TestClassName>>;
    using Reply = CompletionHandler<void(std::optional<WebKit::TestClassName>&&)>;
    explicit TestSynchronousMessage(bool value)
        : m_arguments(value)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<bool> m_arguments;
};

} // namespace TestWithSuperclass
} // namespace Messages
