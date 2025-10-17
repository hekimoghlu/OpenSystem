/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "RemoteCDMInstanceProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "GPUConnectionToWebProcess.h"
#include "RemoteCDMInstanceConfiguration.h"
#include "RemoteCDMInstanceMessages.h"
#include "RemoteCDMInstanceSessionProxy.h"
#include <WebCore/CDMInstance.h>

namespace WebKit {

using namespace WebCore;

Ref<RemoteCDMInstanceProxy> RemoteCDMInstanceProxy::create(RemoteCDMProxy& cdm, Ref<CDMInstance>&& priv, RemoteCDMInstanceIdentifier identifier)
{
    auto configuration = makeUniqueRefWithoutFastMallocCheck<RemoteCDMInstanceConfiguration, RemoteCDMInstanceConfiguration&&>({
        priv->keySystem(),
    });
    return adoptRef(*new RemoteCDMInstanceProxy(cdm, WTFMove(priv), WTFMove(configuration), identifier));
}

RemoteCDMInstanceProxy::RemoteCDMInstanceProxy(RemoteCDMProxy& cdm, Ref<CDMInstance>&& priv, UniqueRef<RemoteCDMInstanceConfiguration>&& configuration, RemoteCDMInstanceIdentifier identifier)
    : m_cdm(cdm)
    , m_instance(WTFMove(priv))
    , m_configuration(WTFMove(configuration))
    , m_identifier(identifier)
#if !RELEASE_LOG_DISABLED
    , m_logger(cdm.logger())
    , m_logIdentifier(cdm.logIdentifier())
#endif
{
    protectedInstance()->setClient(*this);
}

RemoteCDMInstanceProxy::~RemoteCDMInstanceProxy()
{
    protectedInstance()->clearClient();
}

void RemoteCDMInstanceProxy::unrequestedInitializationDataReceived(const String& type, Ref<SharedBuffer>&& initData)
{
    if (!m_cdm)
        return;

    RefPtr factory = m_cdm->factory();
    if (!factory)
        return;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return;

    gpuConnectionToWebProcess->protectedConnection()->send(Messages::RemoteCDMInstance::UnrequestedInitializationDataReceived(type, WTFMove(initData)), m_identifier);
}

void RemoteCDMInstanceProxy::initializeWithConfiguration(const WebCore::CDMKeySystemConfiguration& configuration, AllowDistinctiveIdentifiers allowDistinctiveIdentifiers, AllowPersistentState allowPersistentState, CompletionHandler<void(SuccessValue)>&& completion)
{
    protectedInstance()->initializeWithConfiguration(configuration, allowDistinctiveIdentifiers, allowPersistentState, WTFMove(completion));
}

void RemoteCDMInstanceProxy::setServerCertificate(Ref<SharedBuffer>&& certificate, CompletionHandler<void(SuccessValue)>&& completion)
{
    protectedInstance()->setServerCertificate(WTFMove(certificate), WTFMove(completion));
}

void RemoteCDMInstanceProxy::setStorageDirectory(const String& directory)
{
    if (!m_cdm)
        return;

    RefPtr factory = m_cdm->factory();
    if (!factory)
        return;

    auto mediaKeysStorageDirectory = factory->mediaKeysStorageDirectory();
    if (mediaKeysStorageDirectory.isEmpty())
        return;

    if (directory.startsWith(mediaKeysStorageDirectory))
        protectedInstance()->setStorageDirectory(directory);
}

void RemoteCDMInstanceProxy::createSession(uint64_t logIdentifier, CompletionHandler<void(std::optional<RemoteCDMInstanceSessionIdentifier>)>&& completion)
{
    auto privSession = protectedInstance()->createSession();
    if (!privSession || !m_cdm || !m_cdm->factory()) {
        completion(std::nullopt);
        return;
    }

#if !RELEASE_LOG_DISABLED
    privSession->setLogIdentifier(m_logIdentifier);
#endif

    auto identifier = RemoteCDMInstanceSessionIdentifier::generate();
    auto session = RemoteCDMInstanceSessionProxy::create(m_cdm.get(), privSession.releaseNonNull(), logIdentifier, identifier);
    protectedCdm()->protectedFactory()->addSession(identifier, WTFMove(session));
    completion(identifier);
}

Ref<WebCore::CDMInstance> RemoteCDMInstanceProxy::protectedInstance() const
{
    return m_instance;
}

std::optional<SharedPreferencesForWebProcess> RemoteCDMInstanceProxy::sharedPreferencesForWebProcess() const
{
    if (!m_cdm)
        return std::nullopt;

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_cdm->sharedPreferencesForWebProcess();
}

RefPtr<RemoteCDMProxy> RemoteCDMInstanceProxy::protectedCdm() const
{
    return m_cdm.get();
}

}

#endif
