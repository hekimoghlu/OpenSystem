/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#include "NetworkDataTask.h"

#include "AuthenticationManager.h"
#include "NetworkDataTaskBlob.h"
#include "NetworkDataTaskDataURL.h"
#include "NetworkLoadParameters.h"
#include "NetworkProcess.h"
#include "NetworkSession.h"
#include <WebCore/RegistrableDomain.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/RunLoop.h>
#include <wtf/text/MakeString.h>

#if PLATFORM(COCOA)
#include "NetworkDataTaskCocoa.h"
#endif
#if USE(SOUP)
#include "NetworkDataTaskSoup.h"
#endif
#if USE(CURL)
#include "NetworkDataTaskCurl.h"
#endif

namespace WebKit {
using namespace WebCore;

Ref<NetworkDataTask> NetworkDataTask::create(NetworkSession& session, NetworkDataTaskClient& client, const NetworkLoadParameters& parameters)
{
    ASSERT(!parameters.request.url().protocolIsBlob());
    auto dataTask = [&] {
#if PLATFORM(COCOA)
        return NetworkDataTaskCocoa::create(session, client, parameters);
#else
        if (parameters.request.url().protocolIsData())
            return NetworkDataTaskDataURL::create(session, client, parameters);
#if USE(SOUP)
        return NetworkDataTaskSoup::create(session, client, parameters);
#endif
#if USE(CURL)
        return NetworkDataTaskCurl::create(session, client, parameters);
#endif
#endif
    }();

#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    dataTask->setEmulatedConditions(session.bytesPerSecondLimit());
#endif

    return dataTask;
}

NetworkDataTask::NetworkDataTask(NetworkSession& session, NetworkDataTaskClient& client, const ResourceRequest& requestWithCredentials, StoredCredentialsPolicy storedCredentialsPolicy, bool shouldClearReferrerOnHTTPSToHTTPRedirect, bool dataTaskIsForMainFrameNavigation)
    : m_session(session)
    , m_client(&client)
    , m_partition(requestWithCredentials.cachePartition())
    , m_storedCredentialsPolicy(storedCredentialsPolicy)
    , m_lastHTTPMethod(requestWithCredentials.httpMethod())
    , m_firstRequest(requestWithCredentials)
    , m_shouldClearReferrerOnHTTPSToHTTPRedirect(shouldClearReferrerOnHTTPSToHTTPRedirect)
    , m_dataTaskIsForMainFrameNavigation(dataTaskIsForMainFrameNavigation)
{
    ASSERT(RunLoop::isMain());

    if (!requestWithCredentials.url().isValid()) {
        scheduleFailure(FailureType::InvalidURL);
        return;
    }

    if (!portAllowed(requestWithCredentials.url()) || isIPAddressDisallowed(requestWithCredentials.url())) {
        scheduleFailure(FailureType::Blocked);
        return;
    }

    if (!session.networkProcess().ftpEnabled()
        && requestWithCredentials.url().protocolIsInFTPFamily()) {
        scheduleFailure(FailureType::FTPDisabled);
        return;
    }

    m_session->registerNetworkDataTask(*this);
}

NetworkDataTask::~NetworkDataTask()
{
    ASSERT(RunLoop::isMain());
    ASSERT(!m_client);

    if (m_session)
        m_session->unregisterNetworkDataTask(*this);
}

void NetworkDataTask::scheduleFailure(FailureType type)
{
    m_failureScheduled = true;
    RunLoop::protectedMain()->dispatch([this, weakThis = ThreadSafeWeakPtr { *this }, type] {
        auto protectedThis = weakThis.get();
        if (!protectedThis || !m_client)
            return;

        switch (type) {
        case FailureType::Blocked:
            m_client->wasBlocked();
            return;
        case FailureType::InvalidURL:
            m_client->cannotShowURL();
            return;
        case FailureType::RestrictedURL:
            m_client->wasBlockedByRestrictions();
            return;
        case FailureType::FTPDisabled:
            m_client->wasBlockedByDisabledFTP();
        }
    });
}

void NetworkDataTask::didReceiveInformationalResponse(ResourceResponse&& headers)
{
    if (m_client)
        m_client->didReceiveInformationalResponse(WTFMove(headers));
}

void NetworkDataTask::didReceiveResponse(ResourceResponse&& response, NegotiatedLegacyTLS negotiatedLegacyTLS, PrivateRelayed privateRelayed, std::optional<IPAddress> resolvedIPAddress, ResponseCompletionHandler&& completionHandler)
{
    auto url = response.url();
    if (response.isHTTP09()) {
        std::optional<uint16_t> port = url.port();
        if (port && !WTF::isDefaultPortForProtocol(port.value(), url.protocol())) {
            completionHandler(PolicyAction::Ignore);
            cancel();
            if (m_client)
                m_client->didCompleteWithError({ String(), 0, url, makeString("Cancelled load from '"_s, url.stringCenterEllipsizedToLength(), "' because it is using HTTP/0.9."_s) });
            return;
        }
    }

    auto lastRequest = m_previousRequest.isNull() ? firstRequest() : m_previousRequest;
    auto firstPartyURL = lastRequest.firstPartyForCookies();
    if (!isTopLevelNavigation()
        && firstPartyURL.protocolIs("https"_s) && !SecurityOrigin::isLocalhostAddress(firstPartyURL.host())
        && url.protocolIs("http"_s) && SecurityOrigin::isLocalhostAddress(url.host())) {
        if (resolvedIPAddress && !resolvedIPAddress->isLoopback()) {
            completionHandler(PolicyAction::Ignore);
            cancel();
            if (m_client)
                m_client->didCompleteWithError({ String(), 0, url, makeString("Cancelled load from '"_s, url.stringCenterEllipsizedToLength(), "' because localhost did not resolve to a loopback address."_s) });
            return;
        }
    }

    response.setSource(ResourceResponse::Source::Network);
    if (negotiatedLegacyTLS == NegotiatedLegacyTLS::Yes)
        response.setUsedLegacyTLS(UsedLegacyTLS::Yes);
    if (privateRelayed == PrivateRelayed::Yes)
        response.setWasPrivateRelayed(WasPrivateRelayed::Yes);

    if (m_client)
        m_client->didReceiveResponse(WTFMove(response), negotiatedLegacyTLS, privateRelayed, WTFMove(completionHandler));
    else
        completionHandler(PolicyAction::Ignore);
}

bool NetworkDataTask::shouldCaptureExtraNetworkLoadMetrics() const
{
    return m_client ? m_client->shouldCaptureExtraNetworkLoadMetrics() : false;
}

String NetworkDataTask::description() const
{
    return emptyString();
}

void NetworkDataTask::setH2PingCallback(const URL& url, CompletionHandler<void(Expected<WTF::Seconds, WebCore::ResourceError>&&)>&& completionHandler)
{
    ASSERT_NOT_REACHED();
    completionHandler(makeUnexpected(internalError(url)));
}

PAL::SessionID NetworkDataTask::sessionID() const
{
    return m_session->sessionID();
}

const NetworkSession* NetworkDataTask::networkSession() const
{
    return m_session.get();
}

NetworkSession* NetworkDataTask::networkSession()
{
    return m_session.get();
}

void NetworkDataTask::restrictRequestReferrerToOriginIfNeeded(WebCore::ResourceRequest& request)
{
    if ((m_session->sessionID().isEphemeral() || m_session->isTrackingPreventionEnabled()) && m_session->shouldDowngradeReferrer() && request.isThirdParty())
        request.setExistingHTTPReferrerToOriginString();
}

String NetworkDataTask::attributedBundleIdentifier(WebPageProxyIdentifier pageID)
{
    if (auto* session = networkSession())
        return session->attributedBundleIdentifierFromPageIdentifier(pageID);
    return { };
}

void NetworkDataTask::setPendingDownload(PendingDownload& pendingDownload)
{
    ASSERT(!m_pendingDownload);
    m_pendingDownload = { pendingDownload };
}

PendingDownload* NetworkDataTask::pendingDownload() const
{
    return m_pendingDownload.get();
}

} // namespace WebKit
