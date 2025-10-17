/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#include "NetworkTransportStream.h"

#include <WebCore/Exception.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkTransportStream);

#if !PLATFORM(COCOA)
NetworkTransportStream::NetworkTransportStream()
    : m_identifier(WebCore::WebTransportStreamIdentifier::generate())
    , m_streamType(NetworkTransportStreamType::Bidirectional)
    , m_streamState(NetworkTransportStreamState::Ready)
{
}

void NetworkTransportStream::sendBytes(std::span<const uint8_t>, bool, CompletionHandler<void(std::optional<WebCore::Exception>&&)>&& completionHandler)
{
    completionHandler(std::nullopt);
}

void NetworkTransportStream::cancelReceive(std::optional<WebCore::WebTransportStreamErrorCode>)
{
}

void NetworkTransportStream::cancelSend(std::optional<WebCore::WebTransportStreamErrorCode>)
{
}

void NetworkTransportStream::cancel(std::optional<WebCore::WebTransportStreamErrorCode>)
{
}
#endif

}
