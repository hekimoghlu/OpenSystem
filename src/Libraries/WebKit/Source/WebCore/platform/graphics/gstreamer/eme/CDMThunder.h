/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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

#if ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER)

#include "CDMFactory.h"
#include "CDMInstanceSession.h"
#include "CDMOpenCDMTypes.h"
#include "CDMPrivate.h"
#include "CDMProxy.h"
#include "GStreamerEMEUtilities.h"
#include "MediaKeyStatus.h"
#include "SharedBuffer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

namespace Thunder {

struct ThunderSystemDeleter {
    OpenCDMError operator()(OpenCDMSystem* ptr) const { return opencdm_destruct_system(ptr); }
};

using UniqueThunderSystem = std::unique_ptr<OpenCDMSystem, ThunderSystemDeleter>;

} // namespace Thunder

class CDMFactoryThunder final : public CDMFactory, public CDMProxyFactory {
    WTF_MAKE_TZONE_ALLOCATED(CDMFactoryThunder);
public:
    static CDMFactoryThunder& singleton();

    virtual ~CDMFactoryThunder() = default;

    std::unique_ptr<CDMPrivate> createCDM(const String&, const CDMPrivateClient&) final;
    RefPtr<CDMProxy> createCDMProxy(const String&) final;
    bool supportsKeySystem(const String&) final;
    const Vector<String>& supportedKeySystems() const;

private:
    friend class NeverDestroyed<CDMFactoryThunder>;
    CDMFactoryThunder() = default;
};

class CDMPrivateThunder final : public CDMPrivate {
    WTF_MAKE_TZONE_ALLOCATED(CDMPrivateThunder);
public:
    CDMPrivateThunder(const String& keySystem);
    virtual ~CDMPrivateThunder() = default;

    Vector<AtomString> supportedInitDataTypes() const final;
    bool supportsConfiguration(const CDMKeySystemConfiguration&) const final;
    bool supportsConfigurationWithRestrictions(const CDMKeySystemConfiguration& configuration, const CDMRestrictions&) const final
    {
        return supportsConfiguration(configuration);
    }
    bool supportsSessionTypeWithConfiguration(const CDMSessionType&, const CDMKeySystemConfiguration& configuration) const final
    {
        return supportsConfiguration(configuration);
    }
    Vector<AtomString> supportedRobustnesses() const final;
    CDMRequirement distinctiveIdentifiersRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const final;
    CDMRequirement persistentStateRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const final;
    bool distinctiveIdentifiersAreUniquePerOriginAndClearable(const CDMKeySystemConfiguration&) const final;
    RefPtr<CDMInstance> createInstance() final;
    void loadAndInitialize() final;
    bool supportsServerCertificates() const final;
    bool supportsSessions() const final;
    bool supportsInitData(const AtomString&, const SharedBuffer&) const final;
    RefPtr<SharedBuffer> sanitizeResponse(const SharedBuffer&) const final;
    std::optional<String> sanitizeSessionId(const String&) const final;

private:
    String m_keySystem;
    Thunder::UniqueThunderSystem m_thunderSystem;
};

class CDMInstanceThunder final : public CDMInstanceProxy {
public:
    CDMInstanceThunder(const String& keySystem);
    virtual ~CDMInstanceThunder() = default;

    // CDMInstance
    ImplementationType implementationType() const final { return ImplementationType::Thunder; }
    void initializeWithConfiguration(const CDMKeySystemConfiguration&, AllowDistinctiveIdentifiers, AllowPersistentState, SuccessCallback&&) final;
    void setServerCertificate(Ref<SharedBuffer>&&, SuccessCallback&&) final;
    void setStorageDirectory(const String&) final;
    const String& keySystem() const final { return m_keySystem; }
    RefPtr<CDMInstanceSession> createSession() final;

    OpenCDMSystem& thunderSystem() const { return *m_thunderSystem.get(); };

private:
    Thunder::UniqueThunderSystem m_thunderSystem;
    String m_keySystem;
};

class CDMInstanceSessionThunder final : public CDMInstanceSessionProxy {
public:
    CDMInstanceSessionThunder(CDMInstanceThunder&);

    void requestLicense(LicenseType, KeyGroupingStrategy, const AtomString& initDataType, Ref<SharedBuffer>&& initData, LicenseCallback&&) final;
    void updateLicense(const String&, LicenseType, Ref<SharedBuffer>&&, LicenseUpdateCallback&&) final;
    void loadSession(LicenseType, const String&, const String&, LoadSessionCallback&&) final;
    void closeSession(const String&, CloseSessionCallback&&) final;
    void removeSessionData(const String&, LicenseType, RemoveSessionDataCallback&&) final;
    void storeRecordOfKeyUsage(const String&) final;

    bool isValid() const { return m_session && m_message && !m_message->isEmpty(); }

    void setClient(WeakPtr<CDMInstanceSessionClient>&& client) final { m_client = WTFMove(client); }
    void clearClient() final { m_client.clear(); }

private:
    CDMInstanceThunder* cdmInstanceThunder() const;

    using Notification = void (CDMInstanceSessionThunder::*)(RefPtr<WebCore::SharedBuffer>&&);
    using ChallengeGeneratedCallback = Function<void()>;
    using SessionChangedCallback = Function<void(bool, RefPtr<SharedBuffer>&&)>;

    void challengeGeneratedCallback(RefPtr<SharedBuffer>&&);
    void keyUpdatedCallback(KeyIDType&&);
    void keysUpdateDoneCallback();
    void errorCallback(RefPtr<SharedBuffer>&&);
    CDMInstanceSession::KeyStatus status(const KeyIDType&) const;
    void sessionFailure();

    // FIXME: Check all original uses of these attributes.
    String m_sessionID;
    KeyStore m_keyStore;
    bool m_doesKeyStoreNeedMerging { false };
    InitData m_initData;
    OpenCDMSessionCallbacks m_thunderSessionCallbacks { };
    BoxPtr<OpenCDMSession> m_session;
    RefPtr<SharedBuffer> m_message;
    bool m_needsIndividualization { false };
    Vector<ChallengeGeneratedCallback> m_challengeCallbacks;
    Vector<SessionChangedCallback> m_sessionChangedCallbacks;
    WeakPtr<CDMInstanceSessionClient> m_client;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CDM_INSTANCE(WebCore::CDMInstanceThunder, WebCore::CDMInstance::ImplementationType::Thunder);

#endif // ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER)
