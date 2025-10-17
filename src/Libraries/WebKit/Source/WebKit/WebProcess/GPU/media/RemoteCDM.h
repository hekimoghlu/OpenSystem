/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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

#include "RemoteCDMConfiguration.h"
#include "RemoteCDMFactory.h"
#include "RemoteCDMIdentifier.h"
#include <WebCore/CDMPrivate.h>
#include <wtf/RefCounted.h>

namespace WebKit {

class RemoteCDM final : public WebCore::CDMPrivate {
public:
    static std::unique_ptr<RemoteCDM> create(WeakPtr<RemoteCDMFactory>&&, RemoteCDMIdentifier&&, RemoteCDMConfiguration&&);
    virtual ~RemoteCDM() = default;

private:
    RemoteCDM(WeakPtr<RemoteCDMFactory>&&, RemoteCDMIdentifier&&, RemoteCDMConfiguration&&);

#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(uint64_t) final;
#endif

    void getSupportedConfiguration(WebCore::CDMKeySystemConfiguration&& candidateConfiguration, LocalStorageAccess, SupportedConfigurationCallback&&) final;

    bool supportsConfiguration(const WebCore::CDMKeySystemConfiguration&) const final;
    bool supportsConfigurationWithRestrictions(const WebCore::CDMKeySystemConfiguration&, const WebCore::CDMRestrictions&) const final;
    bool supportsSessionTypeWithConfiguration(const WebCore::CDMSessionType&, const WebCore::CDMKeySystemConfiguration&) const final;
    bool supportsInitData(const AtomString&, const WebCore::SharedBuffer&) const final;
    WebCore::CDMRequirement distinctiveIdentifiersRequirement(const WebCore::CDMKeySystemConfiguration&, const WebCore::CDMRestrictions&) const final;
    WebCore::CDMRequirement persistentStateRequirement(const WebCore::CDMKeySystemConfiguration&, const WebCore::CDMRestrictions&) const final;
    bool distinctiveIdentifiersAreUniquePerOriginAndClearable(const WebCore::CDMKeySystemConfiguration&) const final;
    RefPtr<WebCore::CDMInstance> createInstance() final;
    void loadAndInitialize() final;
    RefPtr<WebCore::SharedBuffer> sanitizeResponse(const WebCore::SharedBuffer&) const final;
    std::optional<String> sanitizeSessionId(const String&) const final;

    Vector<AtomString> supportedInitDataTypes() const final { return m_configuration.supportedInitDataTypes; }
    Vector<AtomString> supportedRobustnesses() const final { return m_configuration.supportedRobustnesses; }
    bool supportsServerCertificates() const final { return m_configuration.supportsServerCertificates; }
    bool supportsSessions() const final { return m_configuration.supportsSessions; }

    WeakPtr<RemoteCDMFactory> m_factory;
    RemoteCDMIdentifier m_identifier;
    RemoteCDMConfiguration m_configuration;
};

}

#endif
