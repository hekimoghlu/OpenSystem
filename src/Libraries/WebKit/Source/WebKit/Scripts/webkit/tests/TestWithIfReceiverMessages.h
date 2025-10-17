/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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

#if ENABLE(APPLE_PAY)

#include "ArgumentCoders.h"
#include "Connection.h"
#include "MessageNames.h"
#include "TestWithIfReceiverMessagesReplies.h"
#include <wtf/Forward.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
struct ApplePayPaymentAuthorizationResult;
}

namespace Messages {
namespace TestWithIfReceiver {

static inline IPC::ReceiverName messageReceiverName()
{
    return IPC::ReceiverName::TestWithIfReceiver;
}

class CompletePaymentSession {
public:
    using Arguments = std::tuple<const WebCore::ApplePayPaymentAuthorizationResult&>;

    static IPC::MessageName name() { return IPC::MessageName::TestWithIfReceiver_CompletePaymentSession; }
    static constexpr bool isSync = false;

    explicit CompletePaymentSession(const WebCore::ApplePayPaymentAuthorizationResult& result)
        : m_arguments(result)
    {
    }

    const Arguments& arguments() const
    {
        return m_arguments;
    }

private:
    Arguments m_arguments;
};

} // namespace TestWithIfReceiver
} // namespace Messages

#endif // ENABLE(APPLE_PAY)
