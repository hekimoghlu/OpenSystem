/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "MessageReceiver.h"
#include "RemoteCDMFactory.h"
#include "RemoteCDMInstanceSessionIdentifier.h"
#include <WebCore/CDMInstanceSession.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SharedBuffer;
}

namespace WebKit {

class RemoteCDMInstanceSession final : public IPC::MessageReceiver, public WebCore::CDMInstanceSession {
public:
    static Ref<RemoteCDMInstanceSession> create(WeakPtr<RemoteCDMFactory>&&, RemoteCDMInstanceSessionIdentifier&&);
    virtual ~RemoteCDMInstanceSession();

    void ref() const final { WebCore::CDMInstanceSession::ref(); }
    void deref() const final { WebCore::CDMInstanceSession::deref(); }

    RemoteCDMInstanceSessionIdentifier identifier() const { return m_identifier; }

private:
    friend class RemoteCDMFactory;
    RemoteCDMInstanceSession(WeakPtr<RemoteCDMFactory>&&, RemoteCDMInstanceSessionIdentifier&&);

#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(uint64_t) final;
#endif

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void updateKeyStatuses(KeyStatusVector&&);
    void sendMessage(WebCore::CDMMessageType, RefPtr<WebCore::SharedBuffer>&&);
    void sessionIdChanged(const String&);

    void setClient(WeakPtr<WebCore::CDMInstanceSessionClient>&& client) final { m_client = WTFMove(client); }
    void clearClient() final { m_client = nullptr; }
    void requestLicense(LicenseType, KeyGroupingStrategy, const AtomString& initDataType, Ref<WebCore::SharedBuffer>&& initData, LicenseCallback&&) final;
    void updateLicense(const String& sessionId, LicenseType, Ref<WebCore::SharedBuffer>&& response, LicenseUpdateCallback&&) final;
    void loadSession(LicenseType, const String& sessionId, const String& origin, LoadSessionCallback&&) final;
    void closeSession(const String& sessionId, CloseSessionCallback&&) final;
    void removeSessionData(const String& sessionId, LicenseType, RemoveSessionDataCallback&&) final;
    void storeRecordOfKeyUsage(const String& sessionId) final;

    RefPtr<RemoteCDMFactory> protectedFactory() const;

    WeakPtr<RemoteCDMFactory> m_factory;
    RemoteCDMInstanceSessionIdentifier m_identifier;
    WeakPtr<WebCore::CDMInstanceSessionClient> m_client;
};

} // namespace WebCore

#endif
