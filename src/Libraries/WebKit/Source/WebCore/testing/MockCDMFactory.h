/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDM.h"
#include "CDMFactory.h"
#include "CDMInstance.h"
#include "CDMInstanceSession.h"
#include "CDMPrivate.h"
#include "MediaKeyEncryptionScheme.h"
#include "MediaKeysRequirement.h"
#include <wtf/HashMap.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class MockCDM;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MockCDM> : std::true_type { };
}

namespace WebCore {

class MockCDMFactory : public RefCountedAndCanMakeWeakPtr<MockCDMFactory>, private CDMFactory {
public:
    static Ref<MockCDMFactory> create() { return adoptRef(*new MockCDMFactory); }
    ~MockCDMFactory();

    const Vector<AtomString>& supportedDataTypes() const { return m_supportedDataTypes; }
    void setSupportedDataTypes(Vector<String>&&);

    const Vector<MediaKeySessionType>& supportedSessionTypes() const { return m_supportedSessionTypes; }
    void setSupportedSessionTypes(Vector<MediaKeySessionType>&& types) { m_supportedSessionTypes = WTFMove(types); }

    const Vector<AtomString>& supportedRobustness() const { return m_supportedRobustness; }
    void setSupportedRobustness(Vector<AtomString>&& supportedRobustness) { m_supportedRobustness = WTFMove(supportedRobustness); }

    MediaKeysRequirement distinctiveIdentifiersRequirement() const { return m_distinctiveIdentifiersRequirement; }
    void setDistinctiveIdentifiersRequirement(MediaKeysRequirement requirement) { m_distinctiveIdentifiersRequirement = requirement; }

    MediaKeysRequirement persistentStateRequirement() const { return m_persistentStateRequirement; }
    void setPersistentStateRequirement(MediaKeysRequirement requirement) { m_persistentStateRequirement = requirement; }

    bool canCreateInstances() const { return m_canCreateInstances; }
    void setCanCreateInstances(bool flag) { m_canCreateInstances = flag; }

    bool supportsServerCertificates() const { return m_supportsServerCertificates; }
    void setSupportsServerCertificates(bool flag) { m_supportsServerCertificates = flag; }

    bool supportsSessions() const { return m_supportsSessions; }
    void setSupportsSessions(bool flag) { m_supportsSessions = flag; }

    const Vector<MediaKeyEncryptionScheme>& supportedEncryptionSchemes() const { return m_supportedEncryptionSchemes; }
    void setSupportedEncryptionSchemes(Vector<MediaKeyEncryptionScheme>&& schemes) { m_supportedEncryptionSchemes = WTFMove(schemes); }

    void unregister();

    bool hasSessionWithID(const String& id);
    void removeSessionWithID(const String& id);
    void addKeysToSessionWithID(const String& id, Vector<Ref<SharedBuffer>>&&);
    const Vector<Ref<SharedBuffer>>* keysForSessionWithID(const String& id) const;
    Vector<Ref<SharedBuffer>> removeKeysFromSessionWithID(const String& id);

private:
    MockCDMFactory();
    std::unique_ptr<CDMPrivate> createCDM(const String&, const CDMPrivateClient&) final;
    bool supportsKeySystem(const String&) final;

    MediaKeysRequirement m_distinctiveIdentifiersRequirement { MediaKeysRequirement::Optional };
    MediaKeysRequirement m_persistentStateRequirement { MediaKeysRequirement::Optional };
    Vector<AtomString> m_supportedDataTypes;
    Vector<MediaKeySessionType> m_supportedSessionTypes;
    Vector<AtomString> m_supportedRobustness;
    Vector<MediaKeyEncryptionScheme> m_supportedEncryptionSchemes;
    bool m_registered { true };
    bool m_canCreateInstances { true };
    bool m_supportsServerCertificates { true };
    bool m_supportsSessions { true };
    HashMap<String, Vector<Ref<SharedBuffer>>> m_sessions;
};

class MockCDM : public CDMPrivate {
    WTF_MAKE_TZONE_ALLOCATED(MockCDM);
public:
    MockCDM(WeakPtr<MockCDMFactory>);

    MockCDMFactory* factory() { return m_factory.get(); }

private:
    friend class MockCDMInstance;

    Vector<AtomString> supportedInitDataTypes() const final;
    Vector<AtomString> supportedRobustnesses() const final;
    bool supportsConfiguration(const MediaKeySystemConfiguration&) const final;
    bool supportsConfigurationWithRestrictions(const MediaKeySystemConfiguration&, const MediaKeysRestrictions&) const final;
    bool supportsSessionTypeWithConfiguration(const MediaKeySessionType&, const MediaKeySystemConfiguration&) const final;
    MediaKeysRequirement distinctiveIdentifiersRequirement(const MediaKeySystemConfiguration&, const MediaKeysRestrictions&) const final;
    MediaKeysRequirement persistentStateRequirement(const MediaKeySystemConfiguration&, const MediaKeysRestrictions&) const final;
    bool distinctiveIdentifiersAreUniquePerOriginAndClearable(const MediaKeySystemConfiguration&) const final;
    RefPtr<CDMInstance> createInstance() final;
    void loadAndInitialize() final;
    bool supportsServerCertificates() const final;
    bool supportsSessions() const final;
    bool supportsInitData(const AtomString&, const SharedBuffer&) const final;
    RefPtr<SharedBuffer> sanitizeResponse(const SharedBuffer&) const final;
    std::optional<String> sanitizeSessionId(const String&) const final;

    WeakPtr<MockCDMFactory> m_factory;
};

class MockCDMInstance : public CDMInstance, public CanMakeWeakPtr<MockCDMInstance> {
public:
    MockCDMInstance(WeakPtr<MockCDM>);

    MockCDMFactory* factory() const { return m_cdm ? m_cdm->factory() : nullptr; }
    bool distinctiveIdentifiersAllowed() const { return m_distinctiveIdentifiersAllowed; }
    bool persistentStateAllowed() const { return m_persistentStateAllowed; }

private:
    ImplementationType implementationType() const final { return ImplementationType::Mock; }
    void initializeWithConfiguration(const MediaKeySystemConfiguration&, AllowDistinctiveIdentifiers, AllowPersistentState, SuccessCallback&&) final;
    void setServerCertificate(Ref<SharedBuffer>&&, SuccessCallback&&) final;
    void setStorageDirectory(const String&) final;
    const String& keySystem() const final;
    RefPtr<CDMInstanceSession> createSession() final;

    WeakPtr<MockCDM> m_cdm;
    bool m_distinctiveIdentifiersAllowed { true };
    bool m_persistentStateAllowed { true };
};

class MockCDMInstanceSession : public CDMInstanceSession {
public:
    MockCDMInstanceSession(WeakPtr<MockCDMInstance>&&);

private:
    void requestLicense(LicenseType, KeyGroupingStrategy, const AtomString& initDataType, Ref<SharedBuffer>&& initData, LicenseCallback&&) final;
    void updateLicense(const String&, LicenseType, Ref<SharedBuffer>&&, LicenseUpdateCallback&&) final;
    void loadSession(LicenseType, const String&, const String&, LoadSessionCallback&&) final;
    void closeSession(const String&, CloseSessionCallback&&) final;
    void removeSessionData(const String&, LicenseType, RemoveSessionDataCallback&&) final;
    void storeRecordOfKeyUsage(const String&) final;

    WeakPtr<MockCDMInstance> m_instance;
};

}

#endif
