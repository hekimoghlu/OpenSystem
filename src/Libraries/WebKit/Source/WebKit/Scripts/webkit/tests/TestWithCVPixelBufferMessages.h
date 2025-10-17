/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#if PLATFORM(COCOA)
#include <WebCore/CVUtilities.h>
#endif
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/ThreadSafeRefCounted.h>


namespace Messages {
namespace TestWithCVPixelBuffer {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithCVPixelBuffer;
}

#if USE(AVFOUNDATION)
class SendCVPixelBuffer {
public:
    using Arguments = std::tuple<RetainPtr<CVPixelBufferRef>>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithCVPixelBuffer_SendCVPixelBuffer; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    explicit SendCVPixelBuffer(const RetainPtr<CVPixelBufferRef>& s0)
        : m_arguments(s0)
    {
    }

    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<const RetainPtr<CVPixelBufferRef>&> m_arguments;
};
#endif

#if USE(AVFOUNDATION)
class ReceiveCVPixelBuffer {
public:
    using Arguments = std::tuple<>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithCVPixelBuffer_ReceiveCVPixelBuffer; }
    static constexpr bool isSync = false;
    static constexpr bool canDispatchOutOfOrder = false;
    static constexpr bool replyCanDispatchOutOfOrder = false;
    static constexpr bool deferSendingIfSuspended = false;

    static IPC::MessageName asyncMessageReplyName() { return IPC::MessageName::TestWithCVPixelBuffer_ReceiveCVPixelBufferReply; }
    static constexpr auto callbackThread = WTF::CompletionHandlerCallThread::ConstructionThread;
    using ReplyArguments = std::tuple<RetainPtr<CVPixelBufferRef>>;
    using Reply = CompletionHandler<void(RetainPtr<CVPixelBufferRef>&&)>;
    using Promise = WTF::NativePromise<RetainPtr<CVPixelBufferRef>, IPC::Error>;
    auto&& arguments()
    {
        return WTFMove(m_arguments);
    }

private:
    std::tuple<> m_arguments;
};
#endif

} // namespace TestWithCVPixelBuffer
} // namespace Messages
