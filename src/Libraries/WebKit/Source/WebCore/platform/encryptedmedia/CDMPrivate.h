/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

#include "CDMInstance.h"
#include "CDMRequirement.h"
#include "CDMSessionType.h"
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class CDMPrivate;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::CDMPrivate> : std::true_type { };
}

#if !RELEASE_LOG_DISABLED
namespace WTF {
class Logger;
}
#endif

namespace WebCore {

struct CDMKeySystemConfiguration;
struct CDMMediaCapability;
struct CDMRestrictions;
class SharedBuffer;

class CDMPrivateClient {
public:
    virtual ~CDMPrivateClient() = default;

#if !RELEASE_LOG_DISABLED
    virtual const Logger& logger() const = 0;
#endif
};

class CDMPrivate : public CanMakeWeakPtr<CDMPrivate> {
public:
    WEBCORE_EXPORT virtual ~CDMPrivate();

#if !RELEASE_LOG_DISABLED
    virtual void setLogIdentifier(uint64_t) { };
#endif

    enum class LocalStorageAccess : bool {
        NotAllowed,
        Allowed,
    };

    using SupportedConfigurationCallback = Function<void(std::optional<CDMKeySystemConfiguration>)>;
    WEBCORE_EXPORT virtual void getSupportedConfiguration(CDMKeySystemConfiguration&& candidateConfiguration, LocalStorageAccess, SupportedConfigurationCallback&&);

    virtual Vector<AtomString> supportedInitDataTypes() const = 0;
    virtual bool supportsConfiguration(const CDMKeySystemConfiguration&) const = 0;
    virtual bool supportsConfigurationWithRestrictions(const CDMKeySystemConfiguration&, const CDMRestrictions&) const = 0;
    virtual bool supportsSessionTypeWithConfiguration(const CDMSessionType&, const CDMKeySystemConfiguration&) const = 0;
    virtual Vector<AtomString> supportedRobustnesses() const = 0;
    virtual CDMRequirement distinctiveIdentifiersRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const = 0;
    virtual CDMRequirement persistentStateRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const = 0;
    virtual bool distinctiveIdentifiersAreUniquePerOriginAndClearable(const CDMKeySystemConfiguration&) const = 0;
    virtual RefPtr<CDMInstance> createInstance() = 0;
    virtual void loadAndInitialize() = 0;
    virtual bool supportsServerCertificates() const = 0;
    virtual bool supportsSessions() const = 0;
    virtual bool supportsInitData(const AtomString&, const SharedBuffer&) const = 0;
    virtual RefPtr<SharedBuffer> sanitizeResponse(const SharedBuffer&) const = 0;
    virtual std::optional<String> sanitizeSessionId(const String&) const = 0;

protected:
    WEBCORE_EXPORT CDMPrivate();
    static bool isPersistentType(CDMSessionType);

    enum class ConfigurationStatus {
        Supported,
        NotSupported,
        ConsentDenied,
    };

    enum class ConsentStatus {
        ConsentDenied,
        InformUser,
        Allowed,
    };

    enum class AudioVideoType {
        Audio,
        Video,
    };

    void doSupportedConfigurationStep(CDMKeySystemConfiguration&& candidateConfiguration, CDMRestrictions&&, LocalStorageAccess, SupportedConfigurationCallback&&);
    std::optional<CDMKeySystemConfiguration> getSupportedConfiguration(const CDMKeySystemConfiguration& candidateConfiguration, CDMRestrictions&, LocalStorageAccess);
    std::optional<Vector<CDMMediaCapability>> getSupportedCapabilitiesForAudioVideoType(AudioVideoType, const Vector<CDMMediaCapability>& requestedCapabilities, const CDMKeySystemConfiguration& partialConfiguration, CDMRestrictions&);

    using ConsentStatusCallback = Function<void(ConsentStatus, CDMKeySystemConfiguration&&, CDMRestrictions&&)>;
    void getConsentStatus(CDMKeySystemConfiguration&& accumulatedConfiguration, CDMRestrictions&&, LocalStorageAccess, ConsentStatusCallback&&);
};

}

#endif
