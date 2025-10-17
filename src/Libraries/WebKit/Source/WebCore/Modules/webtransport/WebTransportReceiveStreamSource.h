/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

#include "ReadableStreamSource.h"

namespace WebCore {

class Exception;
class WebTransport;
class WebTransportReceiveStream;

struct WebTransportStreamIdentifierType;

using WebTransportStreamIdentifier = ObjectIdentifier<WebTransportStreamIdentifierType>;

class WebTransportReceiveStreamSource : public RefCountedReadableStreamSource {
public:
    static Ref<WebTransportReceiveStreamSource> createIncomingStreamsSource() { return adoptRef(*new WebTransportReceiveStreamSource()); }
    static Ref<WebTransportReceiveStreamSource> createIncomingDataSource(WebTransport& transport, WebTransportStreamIdentifier identifier) { return adoptRef(*new WebTransportReceiveStreamSource(transport, identifier)); }
    bool receiveIncomingStream(JSC::JSGlobalObject&, Ref<WebTransportReceiveStream>&);
    void receiveBytes(std::span<const uint8_t>, bool, std::optional<WebCore::Exception>&&);
private:
    WebTransportReceiveStreamSource();
    WebTransportReceiveStreamSource(WebTransport&, WebTransportStreamIdentifier);
    void setActive() final { }
    void setInactive() final { }
    void doStart() final { }
    void doPull() final { }
    void doCancel() final;

    bool m_isCancelled { false };
    bool m_isClosed { false };

    ThreadSafeWeakPtr<WebTransport> m_transport;
    const std::optional<WebTransportStreamIdentifier> m_identifier;
};

}
