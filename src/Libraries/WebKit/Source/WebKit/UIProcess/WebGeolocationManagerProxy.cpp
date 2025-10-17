/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "WebGeolocationManagerProxy.h"

#include "APIGeolocationProvider.h"
#include "GeolocationPermissionRequestManagerProxy.h"
#include "GeolocationPermissionRequestProxy.h"
#include "Logging.h"
#include "WebGeolocationManagerMessages.h"
#include "WebGeolocationManagerProxyMessages.h"
#include "WebGeolocationPosition.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"

#define MESSAGE_CHECK(connection, assertion) MESSAGE_CHECK_BASE(assertion, (connection))

namespace WebKit {

ASCIILiteral WebGeolocationManagerProxy::supplementName()
{
    return "WebGeolocationManagerProxy"_s;
}

Ref<WebGeolocationManagerProxy> WebGeolocationManagerProxy::create(WebProcessPool* processPool)
{
    return adoptRef(*new WebGeolocationManagerProxy(processPool));
}

WebGeolocationManagerProxy::WebGeolocationManagerProxy(WebProcessPool* processPool)
    : WebContextSupplement(processPool)
{
    WebContextSupplement::protectedProcessPool()->addMessageReceiver(Messages::WebGeolocationManagerProxy::messageReceiverName(), *this);
}

WebGeolocationManagerProxy::~WebGeolocationManagerProxy() = default;

void WebGeolocationManagerProxy::setProvider(std::unique_ptr<API::GeolocationProvider>&& provider)
{
    m_clientProvider = WTFMove(provider);
}

// WebContextSupplement

void WebGeolocationManagerProxy::processPoolDestroyed()
{
    if (m_perDomainData.isEmpty())
        return;

    m_perDomainData.clear();
    if (m_clientProvider)
        m_clientProvider->stopUpdating(*this);
}

void WebGeolocationManagerProxy::webProcessIsGoingAway(WebProcessProxy& proxy)
{
    Vector<WebCore::RegistrableDomain> affectedDomains;
    for (auto& [registrableDomain, perDomainData] : m_perDomainData) {
        if (perDomainData->watchers.contains(proxy))
            affectedDomains.append(registrableDomain);
    }
    for (auto& registrableDomain : affectedDomains)
        stopUpdatingWithProxy(proxy, registrableDomain);
}

std::optional<SharedPreferencesForWebProcess> WebGeolocationManagerProxy::sharedPreferencesForWebProcess(IPC::Connection& connection) const
{
    RefPtr process = WebProcessProxy::processForConnection(connection);
    return process ? process->sharedPreferencesForWebProcess() : std::nullopt;
}

void WebGeolocationManagerProxy::refWebContextSupplement()
{
    API::Object::ref();
}

void WebGeolocationManagerProxy::derefWebContextSupplement()
{
    API::Object::deref();
}

void WebGeolocationManagerProxy::providerDidChangePosition(WebGeolocationPosition* position)
{
    for (auto& [registrableDomain, perDomainData] : m_perDomainData) {
        perDomainData->lastPosition = position->corePosition();
        for (Ref process : perDomainData->watchers)
            process->send(Messages::WebGeolocationManager::DidChangePosition(registrableDomain, perDomainData->lastPosition.value()), 0);
    }
}

void WebGeolocationManagerProxy::providerDidFailToDeterminePosition(const String& errorMessage)
{
    for (auto& [registrableDomain, perDomainData] : m_perDomainData) {
        for (Ref proxy : perDomainData->watchers)
            proxy->send(Messages::WebGeolocationManager::DidFailToDeterminePosition(registrableDomain, errorMessage), 0);
    }
}

#if PLATFORM(IOS_FAMILY)
void WebGeolocationManagerProxy::resetPermissions()
{
    ASSERT(m_clientProvider);
    for (auto& [registrableDomain, perDomainData] : m_perDomainData) {
        for (Ref proxy : perDomainData->watchers)
            proxy->send(Messages::WebGeolocationManager::ResetPermissions(registrableDomain), 0);
    }
}
#endif

void WebGeolocationManagerProxy::startUpdating(IPC::Connection& connection, const WebCore::RegistrableDomain& registrableDomain, WebPageProxyIdentifier pageProxyID, const String& authorizationToken, bool enableHighAccuracy)
{
    if (RefPtr process = WebProcessProxy::processForConnection(connection))
        startUpdatingWithProxy(*process, registrableDomain, pageProxyID, authorizationToken, enableHighAccuracy);
}

void WebGeolocationManagerProxy::startUpdatingWithProxy(WebProcessProxy& proxy, const WebCore::RegistrableDomain& registrableDomain, WebPageProxyIdentifier pageProxyID, const String& authorizationToken, bool enableHighAccuracy)
{
    RefPtr page = WebProcessProxy::webPage(pageProxyID);
    MESSAGE_CHECK(proxy.connection(), !!page);

    auto isValidAuthorizationToken = page->protectedGeolocationPermissionRequestManager()->isValidAuthorizationToken(authorizationToken);
    MESSAGE_CHECK(proxy.connection(), isValidAuthorizationToken);

    auto& perDomainData = *m_perDomainData.ensure(registrableDomain, [] {
        return makeUnique<PerDomainData>();
    }).iterator->value;

    bool wasUpdating = isUpdating(perDomainData);
    bool highAccuracyWasEnabled = isHighAccuracyEnabled(perDomainData);

    perDomainData.watchers.add(proxy);
    if (enableHighAccuracy)
        perDomainData.watchersNeedingHighAccuracy.add(proxy);

    if (!wasUpdating) {
        providerStartUpdating(perDomainData, registrableDomain);
        return;
    }
    if (!highAccuracyWasEnabled && enableHighAccuracy)
        providerSetEnabledHighAccuracy(perDomainData, enableHighAccuracy);

    if (perDomainData.lastPosition)
        proxy.send(Messages::WebGeolocationManager::DidChangePosition(registrableDomain, perDomainData.lastPosition.value()), 0);
}

void WebGeolocationManagerProxy::stopUpdating(IPC::Connection& connection, const WebCore::RegistrableDomain& registrableDomain)
{
    if (RefPtr process = WebProcessProxy::processForConnection(connection))
        stopUpdatingWithProxy(*process, registrableDomain);
}

void WebGeolocationManagerProxy::stopUpdatingWithProxy(WebProcessProxy& proxy, const WebCore::RegistrableDomain& registrableDomain)
{
    auto it = m_perDomainData.find(registrableDomain);
    if (it == m_perDomainData.end())
        return;

    auto& perDomainData = *it->value;
    bool wasUpdating = isUpdating(perDomainData);
    bool highAccuracyWasEnabled = isHighAccuracyEnabled(perDomainData);

    perDomainData.watchers.remove(proxy);
    perDomainData.watchersNeedingHighAccuracy.remove(proxy);

    if (wasUpdating && !isUpdating(perDomainData))
        providerStopUpdating(perDomainData);
    else {
        bool highAccuracyShouldBeEnabled = isHighAccuracyEnabled(perDomainData);
        if (highAccuracyShouldBeEnabled != highAccuracyWasEnabled)
            providerSetEnabledHighAccuracy(perDomainData, highAccuracyShouldBeEnabled);
    }

    if (perDomainData.watchers.isEmptyIgnoringNullReferences() && perDomainData.watchersNeedingHighAccuracy.isEmptyIgnoringNullReferences())
        m_perDomainData.remove(it);
}

void WebGeolocationManagerProxy::setEnableHighAccuracy(IPC::Connection& connection, const WebCore::RegistrableDomain& registrableDomain, bool enabled)
{
    if (RefPtr process = WebProcessProxy::processForConnection(connection))
        setEnableHighAccuracyWithProxy(*process, registrableDomain, enabled);
}

void WebGeolocationManagerProxy::setEnableHighAccuracyWithProxy(WebProcessProxy& proxy, const WebCore::RegistrableDomain& registrableDomain, bool enabled)
{
    auto it = m_perDomainData.find(registrableDomain);
    ASSERT(it != m_perDomainData.end());
    if (it == m_perDomainData.end())
        return;

    auto& perDomainData = *it->value;
    bool highAccuracyWasEnabled = isHighAccuracyEnabled(perDomainData);

    if (enabled)
        perDomainData.watchersNeedingHighAccuracy.add(proxy);
    else
        perDomainData.watchersNeedingHighAccuracy.remove(proxy);

    if (isUpdating(perDomainData) && highAccuracyWasEnabled != enabled)
        providerSetEnabledHighAccuracy(perDomainData, enabled);
}

bool WebGeolocationManagerProxy::isUpdating(const PerDomainData& perDomainData) const
{
#if PLATFORM(IOS_FAMILY)
    if (!m_clientProvider)
        return !perDomainData.watchers.isEmptyIgnoringNullReferences();
#else
    UNUSED_PARAM(perDomainData);
#endif

    for (auto& perDomainData : m_perDomainData.values()) {
        if (!perDomainData->watchers.isEmptyIgnoringNullReferences())
            return true;
    }
    return false;
}


bool WebGeolocationManagerProxy::isHighAccuracyEnabled(const PerDomainData& perDomainData) const
{
#if PLATFORM(IOS_FAMILY)
    if (!m_clientProvider)
        return !perDomainData.watchersNeedingHighAccuracy.isEmptyIgnoringNullReferences();
#else
    UNUSED_PARAM(perDomainData);
#endif

    for (auto& data : m_perDomainData.values()) {
        if (!data->watchersNeedingHighAccuracy.isEmptyIgnoringNullReferences())
            return true;
    }
    return false;
}

void WebGeolocationManagerProxy::providerStartUpdating(PerDomainData& perDomainData, const WebCore::RegistrableDomain& registrableDomain)
{
#if PLATFORM(IOS_FAMILY)
    if (!m_clientProvider) {
        ASSERT(!perDomainData.provider);
        perDomainData.provider = makeUnique<WebCore::CoreLocationGeolocationProvider>(registrableDomain, *this);
        perDomainData.provider->setEnableHighAccuracy(!perDomainData.watchersNeedingHighAccuracy.isEmptyIgnoringNullReferences());
        return;
    }
#else
    UNUSED_PARAM(registrableDomain);
    if (!m_clientProvider)
        return;
#endif

    m_clientProvider->setEnableHighAccuracy(*this, isHighAccuracyEnabled(perDomainData));
    m_clientProvider->startUpdating(*this);
}

void WebGeolocationManagerProxy::providerStopUpdating(PerDomainData& perDomainData)
{
#if PLATFORM(IOS_FAMILY)
    if (!m_clientProvider) {
        perDomainData.provider = nullptr;
        return;
    }
#else
    UNUSED_PARAM(perDomainData);
    if (!m_clientProvider)
        return;
#endif

    m_clientProvider->stopUpdating(*this);
}

void WebGeolocationManagerProxy::providerSetEnabledHighAccuracy(PerDomainData& perDomainData, bool enabled)
{
#if PLATFORM(IOS_FAMILY)
    if (!m_clientProvider) {
        perDomainData.provider->setEnableHighAccuracy(enabled);
        return;
    }
#else
    UNUSED_PARAM(perDomainData);
    if (!m_clientProvider)
        return;
#endif

    m_clientProvider->setEnableHighAccuracy(*this, enabled);
}

} // namespace WebKit

#undef MESSAGE_CHECK
