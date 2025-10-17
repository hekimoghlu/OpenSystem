/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#include "RemoteCDMProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "RemoteCDMConfiguration.h"
#include "RemoteCDMInstanceConfiguration.h"
#include "RemoteCDMInstanceProxy.h"
#include <WebCore/CDMKeySystemConfiguration.h>
#include <WebCore/CDMPrivate.h>
#include <WebCore/SharedBuffer.h>

namespace WebKit {

using namespace WebCore;

RefPtr<RemoteCDMProxy> RemoteCDMProxy::create(RemoteCDMFactoryProxy& factory, std::unique_ptr<WebCore::CDMPrivate>&& priv)
{
    if (!priv)
        return nullptr;

    auto configuration = makeUniqueRefWithoutFastMallocCheck<RemoteCDMConfiguration, RemoteCDMConfiguration&&>({
        priv->supportedInitDataTypes(),
        priv->supportedRobustnesses(),
        priv->supportsServerCertificates(),
        priv->supportsSessions()
    });

    return adoptRef(new RemoteCDMProxy(factory, WTFMove(priv), WTFMove(configuration)));
}

RemoteCDMProxy::RemoteCDMProxy(RemoteCDMFactoryProxy& factory, std::unique_ptr<CDMPrivate>&& priv, UniqueRef<RemoteCDMConfiguration>&& configuration)
    : m_factory(factory)
    , m_private(WTFMove(priv))
    , m_configuration(WTFMove(configuration))
#if !RELEASE_LOG_DISABLED
    , m_logger(factory.logger())
#endif
{
}

RemoteCDMProxy::~RemoteCDMProxy() = default;

bool RemoteCDMProxy::supportsInitData(const AtomString& type, const SharedBuffer& data)
{
    return m_private->supportsInitData(type, data);
}

RefPtr<SharedBuffer> RemoteCDMProxy::sanitizeResponse(const SharedBuffer& response)
{
    return m_private->sanitizeResponse(response);
}

std::optional<String> RemoteCDMProxy::sanitizeSessionId(const String& sessionId)
{
    return m_private->sanitizeSessionId(sessionId);
}

void RemoteCDMProxy::getSupportedConfiguration(WebCore::CDMKeySystemConfiguration&& configuration, WebCore::CDMPrivate::LocalStorageAccess access, CompletionHandler<void(std::optional<WebCore::CDMKeySystemConfiguration>)>&& callback)
{
    m_private->getSupportedConfiguration(WTFMove(configuration), access, WTFMove(callback));
}

void RemoteCDMProxy::createInstance(CompletionHandler<void(std::optional<RemoteCDMInstanceIdentifier>, RemoteCDMInstanceConfiguration&&)>&& completion)
{
    auto privateInstance = m_private->createInstance();
    if (!privateInstance || !m_factory) {
        completion(std::nullopt, { });
        return;
    }
    auto identifier = RemoteCDMInstanceIdentifier::generate();
    auto instance = RemoteCDMInstanceProxy::create(*this, privateInstance.releaseNonNull(), identifier);
    RemoteCDMInstanceConfiguration configuration = instance->configuration();
    protectedFactory()->addInstance(identifier, WTFMove(instance));
    completion(identifier, WTFMove(configuration));
}

void RemoteCDMProxy::loadAndInitialize()
{
    m_private->loadAndInitialize();
}

void RemoteCDMProxy::setLogIdentifier(uint64_t logIdentifier)
{
#if !RELEASE_LOG_DISABLED
    m_logIdentifier = logIdentifier;
    if (m_factory)
        m_private->setLogIdentifier(m_logIdentifier);
#else
    UNUSED_PARAM(logIdentifier);
#endif
}

std::optional<SharedPreferencesForWebProcess> RemoteCDMProxy::sharedPreferencesForWebProcess() const
{
    if (!m_factory)
        return std::nullopt;

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_factory->sharedPreferencesForWebProcess();
}

}

#endif
