/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "RemoteCDM.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "RemoteCDMInstance.h"
#include "RemoteCDMInstanceConfiguration.h"
#include "RemoteCDMInstanceIdentifier.h"
#include "RemoteCDMProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/CDMKeySystemConfiguration.h>
#include <WebCore/CDMRestrictions.h>
#include <WebCore/SharedBuffer.h>

namespace WebKit {

using namespace WebCore;

std::unique_ptr<RemoteCDM> RemoteCDM::create(WeakPtr<RemoteCDMFactory>&& factory, RemoteCDMIdentifier&& identifier, RemoteCDMConfiguration&& configuration)
{
    return std::unique_ptr<RemoteCDM>(new RemoteCDM(WTFMove(factory), WTFMove(identifier), WTFMove(configuration)));
}

RemoteCDM::RemoteCDM(WeakPtr<RemoteCDMFactory>&& factory, RemoteCDMIdentifier&& identifier, RemoteCDMConfiguration&& configuration)
    : m_factory(WTFMove(factory))
    , m_identifier(WTFMove(identifier))
    , m_configuration(WTFMove(configuration))
{
}

#if !RELEASE_LOG_DISABLED
void RemoteCDM::setLogIdentifier(uint64_t logIdentifier)
{
    if (RefPtr factory = m_factory.get())
        factory->gpuProcessConnection().connection().send(Messages::RemoteCDMProxy::SetLogIdentifier(logIdentifier), m_identifier);
}
#endif

void RemoteCDM::getSupportedConfiguration(CDMKeySystemConfiguration&& configuration, LocalStorageAccess access, SupportedConfigurationCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(std::nullopt);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMProxy::GetSupportedConfiguration(WTFMove(configuration), access), WTFMove(callback), m_identifier);
}

bool RemoteCDM::supportsConfiguration(const CDMKeySystemConfiguration&) const
{
    ASSERT_NOT_REACHED();
    return false;
}

bool RemoteCDM::supportsConfigurationWithRestrictions(const CDMKeySystemConfiguration&, const CDMRestrictions&) const
{
    ASSERT_NOT_REACHED();
    return false;
}

bool RemoteCDM::supportsSessionTypeWithConfiguration(const CDMSessionType&, const CDMKeySystemConfiguration&) const
{
    ASSERT_NOT_REACHED();
    return false;
}

bool RemoteCDM::supportsInitData(const AtomString& type, const SharedBuffer& data) const
{
    // This check will be done, later, inside RemoteCDMInstanceSessionProxy::requestLicense().
    return true;
}

CDMRequirement RemoteCDM::distinctiveIdentifiersRequirement(const CDMKeySystemConfiguration& configuration, const CDMRestrictions& restrictions) const
{
    ASSERT_NOT_REACHED();
    return CDMRequirement::NotAllowed;
}

CDMRequirement RemoteCDM::persistentStateRequirement(const CDMKeySystemConfiguration&, const CDMRestrictions&) const
{
    ASSERT_NOT_REACHED();
    return CDMRequirement::NotAllowed;
}

bool RemoteCDM::distinctiveIdentifiersAreUniquePerOriginAndClearable(const CDMKeySystemConfiguration&) const
{
    ASSERT_NOT_REACHED();
    return false;
}

RefPtr<CDMInstance> RemoteCDM::createInstance()
{
    RefPtr factory = m_factory.get();
    if (!factory)
        return nullptr;

    auto sendResult = factory->gpuProcessConnection().connection().sendSync(Messages::RemoteCDMProxy::CreateInstance(), m_identifier);
    auto [identifier, configuration] = sendResult.takeReplyOr(std::nullopt, RemoteCDMInstanceConfiguration { });
    if (!identifier)
        return nullptr;
    return RemoteCDMInstance::create(factory.get(), WTFMove(*identifier), WTFMove(configuration));
}

void RemoteCDM::loadAndInitialize()
{
    if (RefPtr factory = m_factory.get())
        factory->gpuProcessConnection().connection().send(Messages::RemoteCDMProxy::LoadAndInitialize(), m_identifier);
}

RefPtr<SharedBuffer> RemoteCDM::sanitizeResponse(const SharedBuffer& response) const
{
    // This check will be done, later, inside RemoteCDMInstanceSessionProxy::updateLicense().
    return response.makeContiguous();
}

std::optional<String> RemoteCDM::sanitizeSessionId(const String& sessionId) const
{
    // This check will be done, later, inside RemoteCDMInstanceSessionProxy::loadSession().
    return sessionId;
}

}

#endif
