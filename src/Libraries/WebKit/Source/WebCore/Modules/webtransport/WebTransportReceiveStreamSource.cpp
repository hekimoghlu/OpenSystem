/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#include "WebTransportReceiveStreamSource.h"

#include "JSWebTransportReceiveStream.h"
#include "WebTransport.h"
#include "WebTransportSession.h"

namespace WebCore {

WebTransportReceiveStreamSource::WebTransportReceiveStreamSource()
{
}

WebTransportReceiveStreamSource::WebTransportReceiveStreamSource(WebTransport& transport, WebTransportStreamIdentifier identifier)
    : m_transport(transport)
    , m_identifier(identifier)
{
}

bool WebTransportReceiveStreamSource::receiveIncomingStream(JSC::JSGlobalObject& globalObject, Ref<WebTransportReceiveStream>& stream)
{
    if (m_isCancelled || m_identifier)
        return false;
    auto& jsDOMGlobalObject = *JSC::jsCast<JSDOMGlobalObject*>(&globalObject);
    Locker<JSC::JSLock> locker(jsDOMGlobalObject.vm().apiLock());
    auto value = toJS(&globalObject, &jsDOMGlobalObject, stream.get());
    if (!controller().enqueue(value)) {
        doCancel();
        return false;
    }
    return true;
}

void WebTransportReceiveStreamSource::receiveBytes(std::span<const uint8_t> bytes, bool withFin, std::optional<WebCore::Exception>&& exception)
{
    if (m_isCancelled || m_isClosed || !m_identifier)
        return;
    if (exception) {
        controller().error(*exception);
        clean();
        return;
    }
    auto arrayBuffer = ArrayBuffer::tryCreateUninitialized(bytes.size(), 1);
    if (arrayBuffer)
        memcpySpan(arrayBuffer->mutableSpan(), bytes);
    if (!controller().enqueue(WTFMove(arrayBuffer)))
        doCancel();
    if (withFin) {
        m_isClosed = true;
        controller().close();
        clean();
    }
}

void WebTransportReceiveStreamSource::doCancel()
{
    if (m_isCancelled)
        return;
    m_isCancelled = true;
    if (!m_identifier)
        return;
    RefPtr transport = m_transport.get();
    if (!transport)
        return;
    RefPtr session = transport->session();
    if (!session)
        return;
    // FIXME: Use error code from WebTransportError
    session->cancelReceiveStream(*m_identifier, std::nullopt);
}
}
