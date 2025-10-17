/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#import "VideoReceiverEndpointMessage.h"

#if ENABLE(LINEAR_MEDIA_PLAYER)

#import "XPCEndpoint.h"
#import <wtf/text/ASCIILiteral.h>

namespace WebKit {

static constexpr ASCIILiteral mediaElementIdentifierKey = "video-receiver-media-element-identifier"_s;
static constexpr ASCIILiteral sourceMediaElementIdentifierKey = "video-receiver-source-media-element-identifier"_s;
static constexpr ASCIILiteral sourcePlayerIdentifierKey = "video-receiver-source-player-identifier"_s;
static constexpr ASCIILiteral destinationMediaElementIdentifierKey = "video-receiver-destination-media-element-identifier"_s;
static constexpr ASCIILiteral destinationPlayerIdentifierKey = "video-receiver-destination-player-identifier"_s;
static constexpr ASCIILiteral processIdentifierKey = "video-receiver-process-identifier"_s;
static constexpr ASCIILiteral playerIdentifierKey = "video-receiver-player-identifier"_s;
static constexpr ASCIILiteral endpointKey = "video-receiver-endpoint"_s;
static constexpr ASCIILiteral endpointIdentifierKey = "video-receiver-endpoint-identifier"_s;
static constexpr ASCIILiteral cacheCommandKey = "video-receiver-cache-command"_s;

VideoReceiverEndpointMessage::VideoReceiverEndpointMessage(std::optional<WebCore::ProcessIdentifier> processIdentifier, WebCore::HTMLMediaElementIdentifier mediaElementIdentifier, std::optional<WebCore::MediaPlayerIdentifier> playerIdentifier, WebCore::VideoReceiverEndpoint endpoint, WebCore::VideoReceiverEndpointIdentifier endpointIdentifier)
    : m_processIdentifier { WTFMove(processIdentifier) }
    , m_mediaElementIdentifier { WTFMove(mediaElementIdentifier) }
    , m_playerIdentifier { WTFMove(playerIdentifier) }
    , m_endpoint { WTFMove(endpoint) }
    , m_endpointIdentifier { WTFMove(endpointIdentifier) }
{
    RELEASE_ASSERT(!m_endpoint || xpc_get_type(m_endpoint.get()) == XPC_TYPE_DICTIONARY);
}

VideoReceiverEndpointMessage VideoReceiverEndpointMessage::decode(xpc_object_t message)
{
    auto processIdentifier = xpc_dictionary_get_uint64(message, processIdentifierKey.characters());
    auto mediaElementIdentifier = xpc_dictionary_get_uint64(message, mediaElementIdentifierKey.characters());
    auto playerIdentifier = xpc_dictionary_get_uint64(message, playerIdentifierKey.characters());
    auto endpoint = xpc_dictionary_get_value(message, endpointKey.characters());
    auto endpointIdentifier = xpc_dictionary_get_uint64(message, endpointIdentifierKey.characters());

    return {
        processIdentifier ? std::optional { WebCore::ProcessIdentifier(processIdentifier) } : std::nullopt,
        WebCore::HTMLMediaElementIdentifier(mediaElementIdentifier),
        playerIdentifier ? std::optional { WebCore::MediaPlayerIdentifier(playerIdentifier) } : std::nullopt,
        WebCore::VideoReceiverEndpoint(endpoint),
        WebCore::VideoReceiverEndpointIdentifier(endpointIdentifier),
    };
}

OSObjectPtr<xpc_object_t> VideoReceiverEndpointMessage::encode() const
{
    OSObjectPtr message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, messageName().characters());
    xpc_dictionary_set_uint64(message.get(), processIdentifierKey.characters(), m_processIdentifier ? m_processIdentifier->toUInt64() : 0);
    xpc_dictionary_set_uint64(message.get(), mediaElementIdentifierKey.characters(), m_mediaElementIdentifier.toUInt64());
    xpc_dictionary_set_uint64(message.get(), playerIdentifierKey.characters(), m_playerIdentifier ? m_playerIdentifier->toUInt64() : 0);
    xpc_dictionary_set_value(message.get(), endpointKey.characters(), m_endpoint.get());
    xpc_dictionary_set_uint64(message.get(), endpointIdentifierKey.characters(), m_endpointIdentifier.toUInt64());
    return message;
}

VideoReceiverSwapEndpointsMessage::VideoReceiverSwapEndpointsMessage(std::optional<WebCore::ProcessIdentifier> processIdentifier, WebCore::HTMLMediaElementIdentifier sourceMediaElementIdentifier, std::optional<WebCore::MediaPlayerIdentifier> sourcePlayerIdentifier, WebCore::HTMLMediaElementIdentifier destinationMediaElementIdentifier, std::optional<WebCore::MediaPlayerIdentifier> destinationPlayerIdentifier)
    : m_processIdentifier { WTFMove(processIdentifier) }
    , m_sourceMediaElementIdentifier { WTFMove(sourceMediaElementIdentifier) }
    , m_sourcePlayerIdentifier { WTFMove(sourcePlayerIdentifier) }
    , m_destinationMediaElementIdentifier { WTFMove(destinationMediaElementIdentifier) }
    , m_destinationPlayerIdentifier { WTFMove(destinationPlayerIdentifier) }
{
}

VideoReceiverSwapEndpointsMessage VideoReceiverSwapEndpointsMessage::decode(xpc_object_t message)
{
    auto processIdentifier = xpc_dictionary_get_uint64(message, processIdentifierKey.characters());
    auto sourceMediaElementIdentifier = xpc_dictionary_get_uint64(message, sourceMediaElementIdentifierKey.characters());
    auto sourcePlayerIdentifier = xpc_dictionary_get_uint64(message, sourcePlayerIdentifierKey.characters());
    auto destinationMediaElementIdentifier = xpc_dictionary_get_uint64(message, destinationMediaElementIdentifierKey.characters());
    auto destinationPlayerIdentifier = xpc_dictionary_get_uint64(message, destinationPlayerIdentifierKey.characters());

    return {
        processIdentifier ? std::optional { WebCore::ProcessIdentifier(processIdentifier) } : std::nullopt,
        WebCore::HTMLMediaElementIdentifier(sourceMediaElementIdentifier),
        sourcePlayerIdentifier ? std::optional { WebCore::MediaPlayerIdentifier(sourcePlayerIdentifier) } : std::nullopt,
        WebCore::HTMLMediaElementIdentifier(destinationMediaElementIdentifier),
        destinationPlayerIdentifier ? std::optional { WebCore::MediaPlayerIdentifier(destinationPlayerIdentifier) } : std::nullopt,
    };
}

OSObjectPtr<xpc_object_t> VideoReceiverSwapEndpointsMessage::encode() const
{
    OSObjectPtr message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(message.get(), XPCEndpoint::xpcMessageNameKey, messageName().characters());
    xpc_dictionary_set_uint64(message.get(), processIdentifierKey.characters(), m_processIdentifier ? m_processIdentifier->toUInt64() : 0);
    xpc_dictionary_set_uint64(message.get(), sourceMediaElementIdentifierKey.characters(), m_sourceMediaElementIdentifier.toUInt64());
    xpc_dictionary_set_uint64(message.get(), sourcePlayerIdentifierKey.characters(), m_sourcePlayerIdentifier ? m_sourcePlayerIdentifier->toUInt64() : 0);
    xpc_dictionary_set_uint64(message.get(), destinationMediaElementIdentifierKey.characters(), m_destinationMediaElementIdentifier.toUInt64());
    xpc_dictionary_set_uint64(message.get(), destinationPlayerIdentifierKey.characters(), m_destinationPlayerIdentifier ? m_destinationPlayerIdentifier->toUInt64() : 0);
    return message;
}

} // namespace WebKit

#endif // ENABLE(LINEAR_MEDIA_PLAYER)
