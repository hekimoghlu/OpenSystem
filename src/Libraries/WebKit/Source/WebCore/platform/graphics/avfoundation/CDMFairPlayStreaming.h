/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#include "CDMFactory.h"
#include "CDMPrivate.h"
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVContentKeyRequest;

namespace WebCore {

struct FourCC;

class CDMFactoryFairPlayStreaming final : public CDMFactory {
    WTF_MAKE_TZONE_ALLOCATED(CDMFactoryFairPlayStreaming);
public:
    static CDMFactoryFairPlayStreaming& singleton();

    virtual ~CDMFactoryFairPlayStreaming();

    std::unique_ptr<CDMPrivate> createCDM(const String&, const CDMPrivateClient&) override;
    bool supportsKeySystem(const String&) override;

private:
    friend class NeverDestroyed<CDMFactoryFairPlayStreaming>;
    CDMFactoryFairPlayStreaming();
};

class CDMPrivateFairPlayStreaming final : public CDMPrivate {
    WTF_MAKE_TZONE_ALLOCATED(CDMPrivateFairPlayStreaming);
public:
    CDMPrivateFairPlayStreaming(const CDMPrivateClient&);
    virtual ~CDMPrivateFairPlayStreaming();

#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(uint64_t logIdentifier) final { m_logIdentifier = logIdentifier; }
    const Logger& logger() const { return m_logger; };
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "CDMPrivateFairPlayStreaming"_s; }
#endif

    Vector<AtomString> supportedInitDataTypes() const override;
    bool supportsConfiguration(const CDMKeySystemConfiguration&) const override;
    bool supportsConfigurationWithRestrictions(const CDMKeySystemConfiguration&, const CDMRestrictions&) const override;
    bool supportsSessionTypeWithConfiguration(const CDMSessionType&, const CDMKeySystemConfiguration&) const override;
    Vector<AtomString> supportedRobustnesses() const override;
    CDMRequirement distinctiveIdentifiersRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const override;
    CDMRequirement persistentStateRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const override;
    bool distinctiveIdentifiersAreUniquePerOriginAndClearable(const CDMKeySystemConfiguration&) const override;
    RefPtr<CDMInstance> createInstance() override;
    void loadAndInitialize() override;
    bool supportsServerCertificates() const override;
    bool supportsSessions() const override;
    bool supportsInitData(const AtomString&, const SharedBuffer&) const override;
    RefPtr<SharedBuffer> sanitizeResponse(const SharedBuffer&) const override;
    std::optional<String> sanitizeSessionId(const String&) const override;

    static const AtomString& sinfName();
    static std::optional<Vector<Ref<SharedBuffer>>> extractKeyIDsSinf(const SharedBuffer&);
    static RefPtr<SharedBuffer> sanitizeSinf(const SharedBuffer&);

    static const AtomString& skdName();
    static std::optional<Vector<Ref<SharedBuffer>>> extractKeyIDsSkd(const SharedBuffer&);
    static RefPtr<SharedBuffer> sanitizeSkd(const SharedBuffer&);

#if HAVE(FAIRPLAYSTREAMING_MTPS_INITDATA)
    static const AtomString& mptsName();
    static std::optional<Vector<Ref<SharedBuffer>>> extractKeyIDsMpts(const SharedBuffer&);
    static RefPtr<SharedBuffer> sanitizeMpts(const SharedBuffer&);
    static const Vector<Ref<SharedBuffer>>& mptsKeyIDs();
#endif

    static const Vector<FourCC>& validFairPlayStreamingSchemes();

#if HAVE(AVCONTENTKEYSESSION)
    static Vector<Ref<SharedBuffer>> keyIDsForRequest(AVContentKeyRequest *);
#endif

private:
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

}

#endif // ENABLE(ENCRYPTED_MEDIA)
