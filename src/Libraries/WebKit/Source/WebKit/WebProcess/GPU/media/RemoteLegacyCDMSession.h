/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "MessageReceiver.h"
#include "RemoteLegacyCDMSessionIdentifier.h"
#include <WebCore/LegacyCDMSession.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SharedBuffer;
}

namespace WebKit {

class RemoteLegacyCDMFactory;

class RemoteLegacyCDMSession final
    : public WebCore::LegacyCDMSession
    , public IPC::MessageReceiver
    , public RefCounted<RemoteLegacyCDMSession> {
public:
    static RefPtr<RemoteLegacyCDMSession> create(RemoteLegacyCDMFactory&, RemoteLegacyCDMSessionIdentifier&&, WebCore::LegacyCDMSessionClient&);
    ~RemoteLegacyCDMSession();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    const RemoteLegacyCDMSessionIdentifier& identifier() const { return m_identifier; }

private:
    RemoteLegacyCDMSession(RemoteLegacyCDMFactory&, RemoteLegacyCDMSessionIdentifier&&, WebCore::LegacyCDMSessionClient&);

    // LegacyCDMSession
    void invalidate() final;
    WebCore::LegacyCDMSessionType type() final { return WebCore::CDMSessionTypeRemote; }
    const String& sessionId() const final { return m_sessionId; }
    RefPtr<Uint8Array> generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode) final;
    void releaseKeys() final;
    bool update(Uint8Array*, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode) final;
    RefPtr<ArrayBuffer> cachedKeyForKeyID(const String&) const final;

    // Messages
    void sendMessage(RefPtr<WebCore::SharedBuffer>&& message, const String& destinationURL);
    void sendError(WebCore::LegacyCDMSessionClient::MediaKeyErrorCode, uint32_t systemCode);

    WeakPtr<RemoteLegacyCDMFactory> m_factory;
    RemoteLegacyCDMSessionIdentifier m_identifier;
    WeakPtr<WebCore::LegacyCDMSessionClient> m_client;
    String m_sessionId;
    mutable HashMap<String, RefPtr<ArrayBuffer>> m_cachedKeyCache;
};

}

#endif
