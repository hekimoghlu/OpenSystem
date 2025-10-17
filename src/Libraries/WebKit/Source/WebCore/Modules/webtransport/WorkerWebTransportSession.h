/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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

#include "ScriptExecutionContextIdentifier.h"
#include "WebTransportSession.h"
#include "WebTransportSessionClient.h"

namespace WebCore {

class WebTransport;

class WorkerWebTransportSession : public WebTransportSession, public WebTransportSessionClient {
public:
    static Ref<WorkerWebTransportSession> create(ScriptExecutionContextIdentifier, WebTransportSessionClient&);
    ~WorkerWebTransportSession();

    void ref() const { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    void attachSession(Ref<WebTransportSession>&&);

private:
    WorkerWebTransportSession(ScriptExecutionContextIdentifier, WebTransportSessionClient&);

    void receiveDatagram(std::span<const uint8_t>, bool, std::optional<Exception>&&) final;
    void receiveIncomingUnidirectionalStream(WebTransportStreamIdentifier) final;
    void receiveBidirectionalStream(WebTransportBidirectionalStreamConstructionParameters&&) final;
    void streamReceiveBytes(WebTransportStreamIdentifier, std::span<const uint8_t>, bool, std::optional<Exception>&&) final;
    void networkProcessCrashed() final;

    Ref<WebTransportSendPromise> sendDatagram(std::span<const uint8_t>) final;
    Ref<WritableStreamPromise> createOutgoingUnidirectionalStream() final;
    Ref<BidirectionalStreamPromise> createBidirectionalStream() final;
    void cancelReceiveStream(WebTransportStreamIdentifier, std::optional<WebTransportStreamErrorCode>) final;
    void cancelSendStream(WebTransportStreamIdentifier, std::optional<WebTransportStreamErrorCode>) final;
    void destroyStream(WebTransportStreamIdentifier, std::optional<WebTransportStreamErrorCode>) final;
    void terminate(WebTransportSessionErrorCode, CString&&) final;

    const ScriptExecutionContextIdentifier m_contextID;
    ThreadSafeWeakPtr<WebTransportSessionClient> m_client;
    RefPtr<WebTransportSession> m_session;
};

}
