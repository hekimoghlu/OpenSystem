/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#include "MessagePortChannel.h"
#include "MessagePortChannelProvider.h"
#include "MessagePortIdentifier.h"
#include "ProcessIdentifier.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MessagePortChannelRegistry final : public CanMakeWeakPtr<MessagePortChannelRegistry>, public CanMakeCheckedPtr<MessagePortChannelRegistry> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(MessagePortChannelRegistry, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MessagePortChannelRegistry);
public:
    WEBCORE_EXPORT MessagePortChannelRegistry();

    WEBCORE_EXPORT ~MessagePortChannelRegistry();
    
    WEBCORE_EXPORT void didCreateMessagePortChannel(const MessagePortIdentifier& port1, const MessagePortIdentifier& port2);
    WEBCORE_EXPORT void didEntangleLocalToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote, ProcessIdentifier);
    WEBCORE_EXPORT void didDisentangleMessagePort(const MessagePortIdentifier& local);
    WEBCORE_EXPORT void didCloseMessagePort(const MessagePortIdentifier& local);
    WEBCORE_EXPORT bool didPostMessageToRemote(MessageWithMessagePorts&&, const MessagePortIdentifier& remoteTarget);
    WEBCORE_EXPORT void takeAllMessagesForPort(const MessagePortIdentifier&, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&&);

    WEBCORE_EXPORT MessagePortChannel* existingChannelContainingPort(const MessagePortIdentifier&);

    WEBCORE_EXPORT void messagePortChannelCreated(MessagePortChannel&);
    WEBCORE_EXPORT void messagePortChannelDestroyed(MessagePortChannel&);

private:
    UncheckedKeyHashMap<MessagePortIdentifier, WeakRef<MessagePortChannel>> m_openChannels;
};

} // namespace WebCore
