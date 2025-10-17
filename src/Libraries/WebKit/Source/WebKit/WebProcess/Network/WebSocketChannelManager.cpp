/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#include "WebSocketChannelManager.h"

#include "Decoder.h"
#include <WebCore/WebSocketIdentifier.h>

namespace WebKit {

void WebSocketChannelManager::addChannel(WebSocketChannel& channel)
{
    ASSERT(!m_channels.contains(channel.identifier()));
    m_channels.add(channel.identifier(), channel);
}

void WebSocketChannelManager::networkProcessCrashed()
{
    auto channels = WTFMove(m_channels);
    for (auto& channel : channels.values())
        channel->networkProcessCrashed();
}

void WebSocketChannelManager::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    auto iterator = m_channels.find(AtomicObjectIdentifier<WebCore::WebSocketIdentifierType>(decoder.destinationID()));
    if (iterator != m_channels.end())
        iterator->value->didReceiveMessage(connection, decoder);
}

} // namespace WebKit
