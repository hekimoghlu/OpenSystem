/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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

#include "Connection.h"
#include <WebCore/BroadcastChannelIdentifier.h>
#include <WebCore/ClientOrigin.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
struct MessageWithMessagePorts;
}

namespace WebKit {

class NetworkProcess;

class NetworkBroadcastChannelRegistry : public RefCounted<NetworkBroadcastChannelRegistry> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkBroadcastChannelRegistry);
public:
    static Ref<NetworkBroadcastChannelRegistry> create(NetworkProcess&);
    ~NetworkBroadcastChannelRegistry();

    void removeConnection(IPC::Connection&);

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    void registerChannel(IPC::Connection&, const WebCore::ClientOrigin&, const String& name);
    void unregisterChannel(IPC::Connection&, const WebCore::ClientOrigin&, const String& name);
    void postMessage(IPC::Connection&, const WebCore::ClientOrigin&, const String& name, WebCore::MessageWithMessagePorts&&, CompletionHandler<void()>&&);

private:
    explicit NetworkBroadcastChannelRegistry(NetworkProcess&);

    Ref<NetworkProcess> m_networkProcess;
    using NameToConnectionIdentifiersMap = HashMap<String, Vector<IPC::Connection::UniqueID>>;
    HashMap<WebCore::ClientOrigin, NameToConnectionIdentifiersMap> m_broadcastChannels;
};

} // namespace WebKit
