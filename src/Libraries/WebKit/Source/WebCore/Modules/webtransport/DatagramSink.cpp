/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include "DatagramSink.h"

#include "Exception.h"
#include "WebTransport.h"
#include "WebTransportSession.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

DatagramSink::DatagramSink() = default;

DatagramSink::~DatagramSink() = default;

void DatagramSink::attachTo(WebTransport& transport)
{
    ASSERT(!m_transport.get());
    m_transport = transport;
}

void DatagramSink::write(ScriptExecutionContext& context, JSC::JSValue value, DOMPromiseDeferred<void>&& promise)
{
    if (!context.globalObject())
        return promise.settle(Exception { ExceptionCode::InvalidStateError });

    if (m_isClosed)
        return promise.settle(Exception { ExceptionCode::InvalidStateError });

    auto& globalObject = *JSC::jsCast<JSDOMGlobalObject*>(context.globalObject());
    auto scope = DECLARE_THROW_SCOPE(globalObject.vm());

    auto bufferSource = convert<IDLUnion<IDLArrayBuffer, IDLArrayBufferView>>(globalObject, value);
    if (UNLIKELY(bufferSource.hasException(scope)))
        return promise.settle(Exception { ExceptionCode::ExistingExceptionError });

    RefPtr transport = m_transport.get();
    if (!transport)
        return promise.settle(Exception { ExceptionCode::InvalidStateError });

    RefPtr session = transport->session();
    if (!session)
        return promise.settle(Exception { ExceptionCode::InvalidStateError });

    WTF::switchOn(bufferSource.releaseReturnValue(), [&](auto&& arrayBufferOrView) {
        context.enqueueTaskWhenSettled(session->sendDatagram(arrayBufferOrView->span()), WebCore::TaskSource::Networking, [promise = WTFMove(promise)] (auto&& exception) mutable {
            if (!exception)
                promise.settle(Exception { ExceptionCode::NetworkError });
            else if (*exception)
                promise.settle(WTFMove(**exception));
            else
                promise.resolve();
        });
    });
}

}
