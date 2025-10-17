/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#include "RemoteCDMInstanceSessionProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "GPUConnectionToWebProcess.h"
#include "RemoteCDMFactoryProxy.h"
#include "RemoteCDMInstanceSessionMessages.h"
#include <WebCore/SharedBuffer.h>

namespace WebKit {

using namespace WebCore;

Ref<RemoteCDMInstanceSessionProxy> RemoteCDMInstanceSessionProxy::create(WeakPtr<RemoteCDMProxy>&& proxy, Ref<WebCore::CDMInstanceSession>&& session, uint64_t logIdentifier, RemoteCDMInstanceSessionIdentifier identifier)
{
    Ref sessionProxy = adoptRef(*new RemoteCDMInstanceSessionProxy(WTFMove(proxy), WTFMove(session), logIdentifier, identifier));
    WeakPtr<WebCore::CDMInstanceSessionClient> client = sessionProxy.get();
    sessionProxy->protectedSession()->setClient(WTFMove(client));
    return sessionProxy;
}

RemoteCDMInstanceSessionProxy::RemoteCDMInstanceSessionProxy(WeakPtr<RemoteCDMProxy>&& cdm, Ref<WebCore::CDMInstanceSession>&& session, uint64_t logIdentifier, RemoteCDMInstanceSessionIdentifier identifier)
    : m_cdm(WTFMove(cdm))
    , m_session(WTFMove(session))
    , m_identifier(identifier)
{
}

RemoteCDMInstanceSessionProxy::~RemoteCDMInstanceSessionProxy()
{
}

void RemoteCDMInstanceSessionProxy::setLogIdentifier(uint64_t logIdentifier)
{
#if !RELEASE_LOG_DISABLED
    protectedSession()->setLogIdentifier(logIdentifier);
#else
    UNUSED_PARAM(logIdentifier);
#endif
}

void RemoteCDMInstanceSessionProxy::requestLicense(LicenseType type, KeyGroupingStrategy keyGroupingStrategy, AtomString initDataType, RefPtr<WebCore::SharedBuffer>&& initData, LicenseCallback&& completion)
{
    if (!initData) {
        completion({ }, emptyString(), false, false);
        return;
    }

    // Implement the CDMPrivate::supportsInitData() check here:
    if (!protectedCdm()->supportsInitData(initDataType, *initData)) {
        completion({ }, emptyString(), false, false);
        return;
    }

    protectedSession()->requestLicense(type, keyGroupingStrategy, initDataType, initData.releaseNonNull(), [completion = WTFMove(completion)] (Ref<SharedBuffer>&& message, const String& sessionId, bool needsIndividualization, CDMInstanceSession::SuccessValue succeeded) mutable {
        completion(WTFMove(message), sessionId, needsIndividualization, succeeded == CDMInstanceSession::Succeeded);
    });
}

void RemoteCDMInstanceSessionProxy::updateLicense(String sessionId, LicenseType type, RefPtr<SharedBuffer>&& response, LicenseUpdateCallback&& completion)
{
    if (!response) {
        completion(true, { }, std::nullopt, std::nullopt, false);
        return;
    }

    // Implement the CDMPrivate::sanitizeResponse() check here:
    auto sanitizedResponse = protectedCdm()->sanitizeResponse(*response);
    if (!sanitizedResponse) {
        completion(false, { }, std::nullopt, std::nullopt, false);
        return;
    }

    protectedSession()->updateLicense(sessionId, type, sanitizedResponse.releaseNonNull(), [completion = WTFMove(completion)] (bool sessionClosed, std::optional<CDMInstanceSession::KeyStatusVector>&& keyStatuses, std::optional<double>&& expirationTime, std::optional<CDMInstanceSession::Message>&& message, CDMInstanceSession::SuccessValue succeeded) mutable {
        completion(sessionClosed, WTFMove(keyStatuses), WTFMove(expirationTime), WTFMove(message), succeeded == CDMInstanceSession::Succeeded);
    });
}

void RemoteCDMInstanceSessionProxy::loadSession(LicenseType type, String sessionId, String origin, LoadSessionCallback&& completion)
{
    // Implement the CDMPrivate::sanitizeSessionId() check here:
    auto sanitizedSessionId = protectedCdm()->sanitizeSessionId(sessionId);
    if (!sanitizedSessionId) {
        completion(std::nullopt, std::nullopt, std::nullopt, false, CDMInstanceSession::SessionLoadFailure::MismatchedSessionType);
        return;
    }

    protectedSession()->loadSession(type, *sanitizedSessionId, origin, [completion = WTFMove(completion)] (std::optional<CDMInstanceSession::KeyStatusVector>&& keyStatuses, std::optional<double>&& expirationTime, std::optional<CDMInstanceSession::Message>&& message, CDMInstanceSession::SuccessValue succeeded, CDMInstanceSession::SessionLoadFailure failure) mutable {
        completion(WTFMove(keyStatuses), WTFMove(expirationTime), WTFMove(message), succeeded == CDMInstanceSession::Succeeded, failure);
    });
}

void RemoteCDMInstanceSessionProxy::closeSession(const String& sessionId, CloseSessionCallback&& completion)
{
    protectedSession()->closeSession(sessionId, [completion = WTFMove(completion)] () mutable {
        completion();
    });
}

void RemoteCDMInstanceSessionProxy::removeSessionData(const String& sessionId, LicenseType type, RemoveSessionDataCallback&& completion)
{
    protectedSession()->removeSessionData(sessionId, type, [completion = WTFMove(completion)] (CDMInstanceSession::KeyStatusVector&& keyStatuses, RefPtr<SharedBuffer>&& expiredSessionsData, CDMInstanceSession::SuccessValue succeeded) mutable {
        completion(WTFMove(keyStatuses), WTFMove(expiredSessionsData), succeeded == CDMInstanceSession::Succeeded);
    });
}

void RemoteCDMInstanceSessionProxy::storeRecordOfKeyUsage(const String& sessionId)
{
    protectedSession()->storeRecordOfKeyUsage(sessionId);
}

void RemoteCDMInstanceSessionProxy::updateKeyStatuses(KeyStatusVector&& keyStatuses)
{
    if (!m_cdm)
        return;

    RefPtr factory = m_cdm->factory();
    if (!factory)
        return;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return;

    gpuConnectionToWebProcess->protectedConnection()->send(Messages::RemoteCDMInstanceSession::UpdateKeyStatuses(WTFMove(keyStatuses)), m_identifier);
}

void RemoteCDMInstanceSessionProxy::sendMessage(CDMMessageType type, Ref<SharedBuffer>&& message)
{
    if (!m_cdm)
        return;

    RefPtr factory = m_cdm->factory();
    if (!factory)
        return;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return;

    gpuConnectionToWebProcess->protectedConnection()->send(Messages::RemoteCDMInstanceSession::SendMessage(type, WTFMove(message)), m_identifier);
}

void RemoteCDMInstanceSessionProxy::sessionIdChanged(const String& sessionId)
{
    if (!m_cdm)
        return;

    RefPtr factory = m_cdm->factory();
    if (!factory)
        return;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return;

    gpuConnectionToWebProcess->protectedConnection()->send(Messages::RemoteCDMInstanceSession::SessionIdChanged(sessionId), m_identifier);
}

std::optional<SharedPreferencesForWebProcess> RemoteCDMInstanceSessionProxy::sharedPreferencesForWebProcess() const
{
    if (!m_cdm)
        return std::nullopt;

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_cdm->sharedPreferencesForWebProcess();
}

RefPtr<RemoteCDMProxy> RemoteCDMInstanceSessionProxy::protectedCdm() const
{
    return m_cdm.get();
}

}

#endif
