/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include "APIHTTPCookieStore.h"

#include "NetworkProcessMessages.h"
#include "WebCookieManagerMessages.h"
#include "WebProcessMessages.h"
#include "WebProcessPool.h"
#include "WebsiteDataStore.h"
#include "WebsiteDataStoreParameters.h"
#include <WebCore/Cookie.h>
#include <WebCore/CookieStorage.h>
#include <WebCore/HTTPCookieAcceptPolicy.h>
#include <WebCore/NetworkStorageSession.h>
#include <wtf/CallbackAggregator.h>

#if PLATFORM(IOS_FAMILY)
#include "DefaultWebBrowserChecks.h"
#endif

using namespace WebKit;

namespace API {

HTTPCookieStore::HTTPCookieStore(WebKit::WebsiteDataStore& websiteDataStore)
    : m_sessionID(websiteDataStore.sessionID())
    , m_owningDataStore(websiteDataStore)
{
}

HTTPCookieStore::~HTTPCookieStore()
{
    ASSERT(m_observers.isEmptyIgnoringNullReferences());
}

void HTTPCookieStore::filterAppBoundCookies(Vector<WebCore::Cookie>&& cookies, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&& completionHandler)
{
#if ENABLE(APP_BOUND_DOMAINS)
    if (!m_owningDataStore)
        return completionHandler({ });
    m_owningDataStore->getAppBoundDomains([cookies = WTFMove(cookies), completionHandler = WTFMove(completionHandler)] (auto& domains) mutable {
        Vector<WebCore::Cookie> appBoundCookies;
        if (!domains.isEmpty() && !isFullWebBrowserOrRunningTest()) {
            for (auto& cookie : WTFMove(cookies)) {
                if (domains.contains(WebCore::RegistrableDomain::uncheckedCreateFromHost(cookie.domain)))
                    appBoundCookies.append(WTFMove(cookie));
            }
        } else
            appBoundCookies = WTFMove(cookies);
        completionHandler(WTFMove(appBoundCookies));
    });
#else
    completionHandler(WTFMove(cookies));
#endif
}

void HTTPCookieStore::cookies(CompletionHandler<void(Vector<WebCore::Cookie>&&)>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists()) {
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::GetAllCookies(m_sessionID), [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (Vector<WebCore::Cookie>&& cookies) mutable {
            filterAppBoundCookies(WTFMove(cookies), WTFMove(completionHandler));
        });
    } else
        completionHandler({ });
}

void HTTPCookieStore::cookiesForURL(WTF::URL&& url, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists()) {
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::GetCookies(m_sessionID, url), [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (Vector<WebCore::Cookie>&& cookies) mutable {
            filterAppBoundCookies(WTFMove(cookies), WTFMove(completionHandler));
        });
    } else
        completionHandler({ });
}

void HTTPCookieStore::setCookies(Vector<WebCore::Cookie>&& cookies, CompletionHandler<void()>&& completionHandler)
{
    filterAppBoundCookies(WTFMove(cookies), [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (auto&& appBoundCookies) mutable {
        if (auto* networkProcess = networkProcessLaunchingIfNecessary())
            networkProcess->sendWithAsyncReply(Messages::WebCookieManager::SetCookie(m_sessionID, appBoundCookies), WTFMove(completionHandler));
        else
            completionHandler();
    });
}

void HTTPCookieStore::deleteCookie(const WebCore::Cookie& cookie, CompletionHandler<void()>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::DeleteCookie(m_sessionID, cookie), WTFMove(completionHandler));
    else
        completionHandler();
}

void HTTPCookieStore::deleteAllCookies(CompletionHandler<void()>&& completionHandler)
{
    auto callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));

    if (m_owningDataStore) {
        for (auto& processPool : m_owningDataStore->processPools()) {
            processPool->forEachProcessForSession(m_sessionID, [&](auto& process) {
                if (!process.canSendMessage())
                    return;
                process.sendWithAsyncReply(Messages::WebProcess::DeleteAllCookies(), [callbackAggregator] { });
            });
        }
    }
    if (auto* networkProcess = networkProcessLaunchingIfNecessary())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::DeleteAllCookies(m_sessionID), [callbackAggregator] { });
}

void HTTPCookieStore::deleteCookiesForHostnames(const Vector<WTF::String>& hostnames, CompletionHandler<void()>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::DeleteCookiesForHostnames(m_sessionID, hostnames), WTFMove(completionHandler));
    else
        completionHandler();
}

void HTTPCookieStore::setHTTPCookieAcceptPolicy(WebCore::HTTPCookieAcceptPolicy policy, CompletionHandler<void()>&& completionHandler)
{
    if (auto* networkProcess = networkProcessLaunchingIfNecessary())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::SetHTTPCookieAcceptPolicy(m_sessionID, policy), WTFMove(completionHandler));
    else
        completionHandler();
}

void HTTPCookieStore::getHTTPCookieAcceptPolicy(CompletionHandler<void(const WebCore::HTTPCookieAcceptPolicy&)>&& completionHandler)
{
    if (auto* networkProcess = networkProcessLaunchingIfNecessary())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::GetHTTPCookieAcceptPolicy(m_sessionID), WTFMove(completionHandler));
    else
        completionHandler({ });
}

void HTTPCookieStore::flushCookies(CompletionHandler<void()>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->sendWithAsyncReply(Messages::NetworkProcess::FlushCookies(m_sessionID), WTFMove(completionHandler));
    else
        completionHandler();
}

void HTTPCookieStore::registerObserver(HTTPCookieStoreObserver& observer)
{
    bool wasObserving = !m_observers.isEmptyIgnoringNullReferences();
    m_observers.add(observer);
    if (wasObserving)
        return;

    if (auto* networkProcess = networkProcessLaunchingIfNecessary())
        networkProcess->send(Messages::WebCookieManager::StartObservingCookieChanges(m_sessionID), 0);
}

void HTTPCookieStore::unregisterObserver(HTTPCookieStoreObserver& observer)
{
    m_observers.remove(observer);
    if (!m_observers.isEmptyIgnoringNullReferences())
        return;

    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->send(Messages::WebCookieManager::StopObservingCookieChanges(m_sessionID), 0);
}

void HTTPCookieStore::cookiesDidChange()
{
    for (Ref observer : m_observers)
        observer->cookiesDidChange(*this);
}

WebKit::NetworkProcessProxy* HTTPCookieStore::networkProcessIfExists()
{
    if (!m_owningDataStore)
        return nullptr;
    return m_owningDataStore->networkProcessIfExists();
}

WebKit::NetworkProcessProxy* HTTPCookieStore::networkProcessLaunchingIfNecessary()
{
    if (!m_owningDataStore)
        return nullptr;
    return &m_owningDataStore->networkProcess();
}

} // namespace API
