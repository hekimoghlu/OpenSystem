/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#include "StreamServerConnection.h"
#include <wtf/Forward.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/ThreadSafeRefCounted.h>


namespace Messages {
namespace TestWithStreamServerConnectionHandle {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithStreamServerConnectionHandle;
}

class SendStreamServerConnection {
public:
    using Arguments = std::tuple<IPC::StreamServerConnectionHandle>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithStreamServerConnectionHandle_SendStreamServerConnection; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    explicit SendStreamServerConnection(IPC::StreamServerConnectionHandle&& handle)
        : m_arguments(WTFMove(handle))
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<IPC::StreamServerConnectionHandle&&> m_arguments;
};

} // namespace TestWithStreamServerConnectionHandle
} // namespace Messages
