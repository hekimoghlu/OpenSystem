/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#include "WebTransportSendStreamSink.h"

#include "WebTransportSession.h"
#include <WebCore/Exception.h>
#include <WebCore/IDLTypes.h>
#include <WebCore/JSDOMGlobalObject.h>
#include <wtf/CompletionHandler.h>
#include <wtf/RunLoop.h>

namespace WebKit {

WebTransportSendStreamSink::WebTransportSendStreamSink(WebTransportSession& session, WebCore::WebTransportStreamIdentifier identifier)
    : m_session(session)
    , m_identifier(identifier)
{
    ASSERT(RunLoop::isMain());
}

WebTransportSendStreamSink::~WebTransportSendStreamSink()
{
}

void WebTransportSendStreamSink::write(WebCore::ScriptExecutionContext& context, JSC::JSValue value, WebCore::DOMPromiseDeferred<void>&& promise)
{
    RefPtr session = m_session.get();
    if (!session)
        return promise.reject(WebCore::Exception { WebCore::ExceptionCode::InvalidStateError });

    if (!context.globalObject())
        return promise.reject(WebCore::Exception { WebCore::ExceptionCode::InvalidStateError });

    if (m_isClosed)
        return promise.reject(WebCore::Exception { WebCore::ExceptionCode::InvalidStateError });

    auto& globalObject = *JSC::jsCast<WebCore::JSDOMGlobalObject*>(context.globalObject());
    auto scope = DECLARE_THROW_SCOPE(globalObject.vm());

    auto bufferSource = convert<WebCore::IDLUnion<WebCore::IDLArrayBuffer, WebCore::IDLArrayBufferView>>(globalObject, value);
    if (UNLIKELY(bufferSource.hasException(scope)))
        return promise.settle(WebCore::Exception { WebCore::ExceptionCode::ExistingExceptionError });

    WTF::switchOn(bufferSource.releaseReturnValue(), [&](auto&& arrayBufferOrView) {
        constexpr bool withFin { false };
        context.enqueueTaskWhenSettled(session->streamSendBytes(m_identifier, arrayBufferOrView->span(), withFin), WebCore::TaskSource::Networking, [promise = WTFMove(promise)] (auto&& exception) mutable {
            if (!exception)
                promise.settle(WebCore::Exception { WebCore::ExceptionCode::NetworkError });
            else if (*exception)
                promise.settle(WTFMove(**exception));
            else
                promise.resolve();
        });
    });
}

void WebTransportSendStreamSink::close()
{
    if (m_isClosed)
        return;
    RefPtr session = m_session.get();
    if (session)
        session->streamSendBytes(m_identifier, { }, true);
    m_isClosed = true;
}

void WebTransportSendStreamSink::error(String&&)
{
    if (m_isCancelled)
        return;
    RefPtr session = m_session.get();
    if (session) {
        // FIXME: Use error code from WebTransportError
        session->cancelSendStream(m_identifier, std::nullopt);
    }
    m_isCancelled = true;
}
}
