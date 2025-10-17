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
#import "config.h"
#import "PushServiceConnection.h"

#import <wtf/TZoneMallocInlines.h>
#import <wtf/WorkQueue.h>

namespace WebPushD {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PushServiceConnection);

PushCrypto::ClientKeys PushServiceConnection::generateClientKeys()
{
    return PushCrypto::ClientKeys::generate();
}

void PushServiceConnection::startListeningForPublicToken(Function<void(Vector<uint8_t>&&)>&& handler)
{
    m_publicTokenChangeHandler = WTFMove(handler);

    if (m_pendingPublicToken.isEmpty())
        return;

    m_publicTokenChangeHandler(std::exchange(m_pendingPublicToken, { }));
}

void PushServiceConnection::didReceivePublicToken(Vector<uint8_t>&& token)
{
    if (!m_publicTokenChangeHandler) {
        m_pendingPublicToken = WTFMove(token);
        return;
    }

    m_publicTokenChangeHandler(WTFMove(token));
}

void PushServiceConnection::setPublicTokenForTesting(Vector<uint8_t>&&)
{
}

void PushServiceConnection::startListeningForPushMessages(IncomingPushMessageHandler&& handler)
{
    m_incomingPushMessageHandler = WTFMove(handler);

    if (!m_pendingPushes.size())
        return;

    WorkQueue::main().dispatch([this]() mutable {
        while (m_pendingPushes.size()) {
            @autoreleasepool {
                auto message = m_pendingPushes.takeFirst();
                m_incomingPushMessageHandler(message.first.get(), message.second.get());
            }
        }
    });
}

void PushServiceConnection::didReceivePushMessage(NSString *topic, NSDictionary *userInfo)
{
    if (!m_incomingPushMessageHandler) {
        m_pendingPushes.append({ topic, userInfo });
        return;
    }

    m_incomingPushMessageHandler(topic, userInfo);
}

} // namespace WebPushD
