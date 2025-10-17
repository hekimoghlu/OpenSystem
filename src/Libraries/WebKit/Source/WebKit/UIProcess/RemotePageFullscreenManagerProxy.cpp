/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#include "config.h"
#include "RemotePageFullscreenManagerProxy.h"

#if ENABLE(FULLSCREEN_API)

#include "WebFullScreenManagerProxy.h"
#include "WebFullScreenManagerProxyMessages.h"
#include "WebProcessProxy.h"

namespace WebKit {

Ref<RemotePageFullscreenManagerProxy> RemotePageFullscreenManagerProxy::create(WebCore::PageIdentifier identifier, WebFullScreenManagerProxy* manager, WebProcessProxy& process)
{
    return adoptRef(*new RemotePageFullscreenManagerProxy(identifier, manager, process));
}

RemotePageFullscreenManagerProxy::RemotePageFullscreenManagerProxy(WebCore::PageIdentifier identifier, WebFullScreenManagerProxy* manager, WebProcessProxy& process)
    : m_identifier(identifier)
    , m_manager(manager)
    , m_process(process)
{
    process.addMessageReceiver(Messages::WebFullScreenManagerProxy::messageReceiverName(), m_identifier, *this);
}

RemotePageFullscreenManagerProxy::~RemotePageFullscreenManagerProxy()
{
    m_process->removeMessageReceiver(Messages::WebFullScreenManagerProxy::messageReceiverName(), m_identifier);
}

void RemotePageFullscreenManagerProxy::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (RefPtr manager = m_manager.get())
        manager->didReceiveMessage(connection, decoder);
}

bool RemotePageFullscreenManagerProxy::didReceiveSyncMessage(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& replyEncoder)
{
    if (RefPtr manager = m_manager.get())
        return manager->didReceiveSyncMessage(connection, decoder, replyEncoder);
    return false;
}

}
#endif
