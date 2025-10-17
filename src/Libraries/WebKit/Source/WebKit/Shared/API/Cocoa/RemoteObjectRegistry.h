/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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

#include "MessageReceiver.h"
#include "ProcessThrottler.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakObjCPtr.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS _WKRemoteObjectRegistry;

namespace WebKit {

class RemoteObjectInvocation;
class UserData;
class WebPage;
class WebProcessProxy;

class RemoteObjectRegistry : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteObjectRegistry);
public:
    virtual ~RemoteObjectRegistry();

    virtual void sendInvocation(const RemoteObjectInvocation&);
    void sendReplyBlock(uint64_t replyID, const UserData& blockInvocation);
    void sendUnusedReply(uint64_t replyID);

protected:
    explicit RemoteObjectRegistry(_WKRemoteObjectRegistry *);
    using MessageSender = std::variant<std::reference_wrapper<WebProcessProxy>, std::reference_wrapper<WebPage>>;
private:
    virtual RefPtr<ProcessThrottler::BackgroundActivity> backgroundActivity(ASCIILiteral) { return nullptr; }
    virtual std::optional<MessageSender> messageSender() = 0;
    virtual std::optional<uint64_t> messageDestinationID() = 0;
    template<typename M> void send(M&&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Message handlers
    void invokeMethod(const RemoteObjectInvocation&);
    void callReplyBlock(uint64_t replyID, const UserData& blockInvocation);
    void releaseUnusedReplyBlock(uint64_t replyID);

    WeakObjCPtr<_WKRemoteObjectRegistry> m_remoteObjectRegistry;
    HashMap<uint64_t, RefPtr<ProcessThrottler::BackgroundActivity>> m_pendingReplies;
};

} // namespace WebKit
