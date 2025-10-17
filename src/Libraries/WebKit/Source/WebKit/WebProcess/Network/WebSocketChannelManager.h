/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

#include "WebSocketChannel.h"

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class Document;
class ThreadableWebSocketChannel;
}

namespace WebKit {

class WebSocketChannelManager {
public:
    // Choose a per-process limit that matches Firefox and Tor's global count (200),
    // and Brave's per-process limit (50). Chrome has a global limit of 256, so
    // any compatibility risk with Chrome should be very low.
    static constexpr size_t maximumSocketCount = 200;

    WebSocketChannelManager() = default;

    void networkProcessCrashed();
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    void addChannel(WebSocketChannel&);
    void removeChannel(WebSocketChannel& channel) { m_channels.remove(channel.identifier() ); }

    bool hasReachedSocketLimit() const { return m_channels.size() >= maximumSocketCount; }

private:
    HashMap<WebCore::WebSocketIdentifier, WeakPtr<WebSocketChannel>> m_channels;
};

} // namespace WebKit
