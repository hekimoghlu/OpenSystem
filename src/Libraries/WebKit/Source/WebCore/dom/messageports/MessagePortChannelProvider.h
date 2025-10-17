/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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

#include "ProcessIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Vector.h>

namespace WebCore {
class MessagePortChannelProvider;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MessagePortChannelProvider> : std::true_type { };
}

namespace WebCore {

class ScriptExecutionContext;
struct MessagePortIdentifier;
struct MessageWithMessagePorts;

class MessagePortChannelProvider : public CanMakeWeakPtr<MessagePortChannelProvider> {
public:
    static MessagePortChannelProvider& fromContext(ScriptExecutionContext&);
    static MessagePortChannelProvider& singleton();
    WEBCORE_EXPORT static void setSharedProvider(MessagePortChannelProvider&);

    virtual ~MessagePortChannelProvider() { }

    // Operations that WebProcesses perform
    virtual void createNewMessagePortChannel(const MessagePortIdentifier& local, const MessagePortIdentifier& remote) = 0;
    virtual void entangleLocalPortInThisProcessToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote) = 0;
    virtual void messagePortDisentangled(const MessagePortIdentifier& local) = 0;
    virtual void messagePortClosed(const MessagePortIdentifier& local) = 0;
    
    virtual void takeAllMessagesForPort(const MessagePortIdentifier&, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&&) = 0;

    virtual void postMessageToRemote(MessageWithMessagePorts&&, const MessagePortIdentifier& remoteTarget) = 0;
};

} // namespace WebCore
