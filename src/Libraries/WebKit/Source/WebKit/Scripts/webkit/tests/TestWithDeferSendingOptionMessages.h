/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>


namespace Messages {
namespace TestWithDeferSendingOption {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithDeferSendingOption;
}

class NoOptions {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithDeferSendingOption_NoOptions; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    explicit NoOptions(const String& url)
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

class NoIndices {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithDeferSendingOption_NoIndices; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = true;

    explicit NoIndices(const String& url)
        : m_arguments(url)
    {
    }

    // Not valid to call this after arguments() is called.
    void encodeCoalescingKey(IPC::Encoder&) const
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const String&> m_arguments;
};

class OneIndex {
public:
    using Arguments = std::tuple<String>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithDeferSendingOption_OneIndex; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = true;

    explicit OneIndex(const String& url)
        : m_arguments(url)
    {
    }

    // Not valid to call this after arguments() is called.
    void encodeCoalescingKey(IPC::Encoder& encoder) const
    {
        encoder << std::get<0>(m_arguments);
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const String&> m_arguments;
};

class MultipleIndices {
public:
    using Arguments = std::tuple<String, int, int, int>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithDeferSendingOption_MultipleIndices; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = true;

    MultipleIndices(const String& url, const int& foo, const int& bar, const int& baz)
        : m_arguments(url, foo, bar, baz)
    {
    }

    // Not valid to call this after arguments() is called.
    void encodeCoalescingKey(IPC::Encoder& encoder) const
    {
        encoder << std::get<2>(m_arguments) << std::get<0>(m_arguments) << std::get<1>(m_arguments);
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const String&, const int&, const int&, const int&> m_arguments;
};

} // namespace TestWithDeferSendingOption
} // namespace Messages
