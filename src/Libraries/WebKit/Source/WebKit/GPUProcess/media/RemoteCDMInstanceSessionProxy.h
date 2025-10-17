/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

#include "Connection.h"
#include "MessageReceiver.h"
#include "RemoteCDMInstanceSessionIdentifier.h"
#include "RemoteCDMProxy.h"
#include <WebCore/CDMInstanceSession.h>
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SharedBuffer;
}

namespace WebKit {

class RemoteCDMInstanceSessionProxy final : private IPC::MessageReceiver, private WebCore::CDMInstanceSessionClient, public RefCounted<RemoteCDMInstanceSessionProxy> {
public:
    static Ref<RemoteCDMInstanceSessionProxy> create(WeakPtr<RemoteCDMProxy>&&, Ref<WebCore::CDMInstanceSession>&&, uint64_t logIdentifier, RemoteCDMInstanceSessionIdentifier);
    virtual ~RemoteCDMInstanceSessionProxy();
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    friend class RemoteCDMFactoryProxy;
    RemoteCDMInstanceSessionProxy(WeakPtr<RemoteCDMProxy>&&, Ref<WebCore::CDMInstanceSession>&&, uint64_t logIdentifier, RemoteCDMInstanceSessionIdentifier);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Types
    using KeyGroupingStrategy = WebCore::CDMInstanceSession::KeyGroupingStrategy;
    using LicenseType = WebCore::CDMInstanceSession::LicenseType;
    using KeyStatusVector = WebCore::CDMInstanceSession::KeyStatusVector;
    using Message = WebCore::CDMInstanceSession::Message;
    using SessionLoadFailure = WebCore::CDMInstanceSession::SessionLoadFailure;
    using LicenseCallback = CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&, const String& sessionId, bool, bool)>;
    using LicenseUpdateCallback = CompletionHandler<void(bool, std::optional<KeyStatusVector>&&, std::optional<double>&&, std::optional<Message>&&, bool)>;
    using LoadSessionCallback = CompletionHandler<void(std::optional<KeyStatusVector>&&, std::optional<double>&&, std::optional<Message>&&, bool, SessionLoadFailure)>;
    using CloseSessionCallback = CompletionHandler<void()>;
    using RemoveSessionDataCallback = CompletionHandler<void(KeyStatusVector&&, RefPtr<WebCore::SharedBuffer>&&, bool)>;
    using StoreRecordCallback = CompletionHandler<void()>;

    // Messages
    void setLogIdentifier(uint64_t);
    void requestLicense(LicenseType, KeyGroupingStrategy, AtomString initDataType, RefPtr<WebCore::SharedBuffer>&& initData, LicenseCallback&&);
    void updateLicense(String sessionId, LicenseType, RefPtr<WebCore::SharedBuffer>&& response, LicenseUpdateCallback&&);
    void loadSession(LicenseType, String sessionId, String origin, LoadSessionCallback&&);
    void closeSession(const String& sessionId, CloseSessionCallback&&);
    void removeSessionData(const String& sessionId, LicenseType, RemoveSessionDataCallback&&);
    void storeRecordOfKeyUsage(const String& sessionId);

    // CDMInstanceSessionClient
    void updateKeyStatuses(KeyStatusVector&&) final;
    void sendMessage(WebCore::CDMMessageType, Ref<WebCore::SharedBuffer>&& message) final;
    void sessionIdChanged(const String&) final;
    PlatformDisplayID displayID() final { return m_displayID; }

    RefPtr<RemoteCDMProxy> protectedCdm() const;
    Ref<WebCore::CDMInstanceSession> protectedSession() const { return m_session; }

    WeakPtr<RemoteCDMProxy> m_cdm;
    Ref<WebCore::CDMInstanceSession> m_session;
    RemoteCDMInstanceSessionIdentifier m_identifier;
    PlatformDisplayID m_displayID { 0 };
};

}

#endif
