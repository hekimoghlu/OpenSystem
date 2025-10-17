/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#import "RemoteObjectRegistry.h"

#import "MessageSenderInlines.h"
#import "RemoteObjectInvocation.h"
#import "RemoteObjectRegistryMessages.h"
#import "UserData.h"
#import "WebPage.h"
#import "WebProcessProxy.h"
#import "_WKRemoteObjectRegistryInternal.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteObjectRegistry);

RemoteObjectRegistry::RemoteObjectRegistry(_WKRemoteObjectRegistry *remoteObjectRegistry)
    : m_remoteObjectRegistry(remoteObjectRegistry)
{
}

RemoteObjectRegistry::~RemoteObjectRegistry() = default;

template<typename M> void RemoteObjectRegistry::send(M&& message)
{
    auto messageSender = this->messageSender();
    auto messageDestinationID = this->messageDestinationID();
    if (!messageSender || !messageDestinationID)
        return;

    WTF::switchOn(*messageSender, [&] (auto sender) {
        sender.get().send(WTFMove(message), *messageDestinationID);
    });
}

void RemoteObjectRegistry::sendInvocation(const RemoteObjectInvocation& invocation)
{
    if (auto& replyInfo = invocation.replyInfo()) {
        ASSERT(!m_pendingReplies.contains(replyInfo->replyID));
        m_pendingReplies.add(replyInfo->replyID, backgroundActivity("RemoteObjectRegistry invocation"_s));
    }

    send(Messages::RemoteObjectRegistry::InvokeMethod(invocation));
}

void RemoteObjectRegistry::sendReplyBlock(uint64_t replyID, const UserData& blockInvocation)
{
    send(Messages::RemoteObjectRegistry::CallReplyBlock(replyID, blockInvocation));
}

void RemoteObjectRegistry::sendUnusedReply(uint64_t replyID)
{
    send(Messages::RemoteObjectRegistry::ReleaseUnusedReplyBlock(replyID));
}

void RemoteObjectRegistry::invokeMethod(const RemoteObjectInvocation& invocation)
{
    [m_remoteObjectRegistry _invokeMethod:invocation];
}

void RemoteObjectRegistry::callReplyBlock(uint64_t replyID, const UserData& blockInvocation)
{
    bool wasRemoved = m_pendingReplies.remove(replyID);
    ASSERT_UNUSED(wasRemoved, wasRemoved);

    [m_remoteObjectRegistry _callReplyWithID:replyID blockInvocation:blockInvocation];
}

void RemoteObjectRegistry::releaseUnusedReplyBlock(uint64_t replyID)
{
    bool wasRemoved = m_pendingReplies.remove(replyID);
    ASSERT_UNUSED(wasRemoved, wasRemoved);

    [m_remoteObjectRegistry _releaseReplyWithID:replyID];
}

} // namespace WebKit
