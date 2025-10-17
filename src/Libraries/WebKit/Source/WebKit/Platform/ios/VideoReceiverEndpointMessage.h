/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

#if ENABLE(LINEAR_MEDIA_PLAYER)

#include <WebCore/HTMLMediaElementIdentifier.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/VideoReceiverEndpoint.h>
#include <wtf/Markable.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebKit {

class VideoReceiverEndpointMessage {
public:
    VideoReceiverEndpointMessage(std::optional<WebCore::ProcessIdentifier>, WebCore::HTMLMediaElementIdentifier, std::optional<WebCore::MediaPlayerIdentifier>, WebCore::VideoReceiverEndpoint, WebCore::VideoReceiverEndpointIdentifier);

    static ASCIILiteral messageName() { return "video-receiver-endpoint"_s; }
    static VideoReceiverEndpointMessage decode(xpc_object_t);
    OSObjectPtr<xpc_object_t> encode() const;

    std::optional<WebCore::ProcessIdentifier> processIdentifier() const { return m_processIdentifier; }
    WebCore::HTMLMediaElementIdentifier mediaElementIdentifier() const { return m_mediaElementIdentifier; }
    std::optional<WebCore::MediaPlayerIdentifier> playerIdentifier() const { return m_playerIdentifier; }
    const WebCore::VideoReceiverEndpoint& endpoint() const { return m_endpoint; }
    WebCore::VideoReceiverEndpointIdentifier endpointIdentifier() const { return m_endpointIdentifier; }

private:
    Markable<WebCore::ProcessIdentifier> m_processIdentifier;
    WebCore::HTMLMediaElementIdentifier m_mediaElementIdentifier;
    Markable<WebCore::MediaPlayerIdentifier> m_playerIdentifier;
    WebCore::VideoReceiverEndpoint m_endpoint;
    WebCore::VideoReceiverEndpointIdentifier m_endpointIdentifier;
};

class VideoReceiverSwapEndpointsMessage {
public:
    VideoReceiverSwapEndpointsMessage(std::optional<WebCore::ProcessIdentifier>, WebCore::HTMLMediaElementIdentifier, std::optional<WebCore::MediaPlayerIdentifier>, WebCore::HTMLMediaElementIdentifier, std::optional<WebCore::MediaPlayerIdentifier>);

    static ASCIILiteral messageName() { return "video-receiver-swap-endpoint"_s; }
    static VideoReceiverSwapEndpointsMessage decode(xpc_object_t);
    OSObjectPtr<xpc_object_t> encode() const;

    std::optional<WebCore::ProcessIdentifier> processIdentifier() const { return m_processIdentifier; }
    WebCore::HTMLMediaElementIdentifier sourceMediaElementIdentifier() const { return m_sourceMediaElementIdentifier; }
    std::optional<WebCore::MediaPlayerIdentifier> sourceMediaPlayerIdentifier() const { return m_sourcePlayerIdentifier; }
    WebCore::HTMLMediaElementIdentifier destinationMediaElementIdentifier() const { return m_destinationMediaElementIdentifier; }
    std::optional<WebCore::MediaPlayerIdentifier> destinationMediaPlayerIdentifier() const { return m_destinationPlayerIdentifier; }

private:
    Markable<WebCore::ProcessIdentifier> m_processIdentifier;
    WebCore::HTMLMediaElementIdentifier m_sourceMediaElementIdentifier;
    Markable<WebCore::MediaPlayerIdentifier> m_sourcePlayerIdentifier;
    WebCore::HTMLMediaElementIdentifier m_destinationMediaElementIdentifier;
    Markable<WebCore::MediaPlayerIdentifier> m_destinationPlayerIdentifier;
};


} // namespace WebKit

#endif // ENABLE(LINEAR_MEDIA_PLAYER)
