/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "WebCookieCache.h"

#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "WebCookieJar.h"
#include "WebProcess.h"
#include <wtf/text/StringBuilder.h>

namespace WebKit {

using namespace WebCore;

WebCookieCache::~WebCookieCache() = default;

bool WebCookieCache::isSupported()
{
#if HAVE(COOKIE_CHANGE_LISTENER_API)
    return true;
#else
    return false;
#endif
}

static String cookiesToString(const Vector<WebCore::Cookie>& cookies)
{
    StringBuilder cookiesBuilder;
    for (auto& cookie : cookies) {
        if (cookie.name.isEmpty())
            continue;
        ASSERT(!cookie.httpOnly);
        if (cookie.httpOnly)
            continue;
        if (!cookiesBuilder.isEmpty())
            cookiesBuilder.append("; "_s);
        cookiesBuilder.append(cookie.name);
        cookiesBuilder.append('=');
        cookiesBuilder.append(cookie.value);
    }
    return cookiesBuilder.toString();
}

String WebCookieCache::cookiesForDOM(const URL& firstParty, const SameSiteInfo& sameSiteInfo, const URL& url, FrameIdentifier frameID, PageIdentifier pageID, WebPageProxyIdentifier webPageProxyID, IncludeSecureCookies includeSecureCookies)
{
    bool hasCacheForHost = m_hostsWithInMemoryStorage.contains<StringViewHashTranslator>(url.host());
    if (!hasCacheForHost || cacheMayBeOutOfSync()) {
        auto host = url.host().toString();
#if HAVE(COOKIE_CHANGE_LISTENER_API)
        if (!hasCacheForHost)
            WebProcess::singleton().protectedCookieJar()->addChangeListenerWithAccess(url, firstParty, frameID, pageID, webPageProxyID, *this);
#endif
        auto sendResult = WebProcess::singleton().ensureNetworkProcessConnection().protectedConnection()->sendSync(Messages::NetworkConnectionToWebProcess::DomCookiesForHost(url), 0);
        if (!sendResult.succeeded())
            return { };

        auto& [cookies] = sendResult.reply();

        if (hasCacheForHost)
            return cookiesToString(cookies);

        pruneCacheIfNecessary();
        m_hostsWithInMemoryStorage.add(WTFMove(host));
        for (auto& cookie : cookies)
            inMemoryStorageSession().setCookie(cookie);
    }
    return inMemoryStorageSession().cookiesForDOM(firstParty, sameSiteInfo, url, frameID, pageID, includeSecureCookies, ApplyTrackingPrevention::No, ShouldRelaxThirdPartyCookieBlocking::No).first;
}

void WebCookieCache::setCookiesFromDOM(const URL& firstParty, const SameSiteInfo& sameSiteInfo, const URL& url, FrameIdentifier frameID, PageIdentifier pageID, const String& cookieString, ShouldRelaxThirdPartyCookieBlocking shouldRelaxThirdPartyCookieBlocking)
{
    if (m_hostsWithInMemoryStorage.contains<StringViewHashTranslator>(url.host()))
        inMemoryStorageSession().setCookiesFromDOM(firstParty, sameSiteInfo, url, frameID, pageID, ApplyTrackingPrevention::No, cookieString, shouldRelaxThirdPartyCookieBlocking);
}

PendingCookieUpdateCounter::Token WebCookieCache::willSetCookieFromDOM()
{
    return m_pendingCookieUpdateCounter.count();
}

void WebCookieCache::didSetCookieFromDOM(PendingCookieUpdateCounter::Token, const URL& firstParty, const SameSiteInfo& sameSiteInfo, const URL& url, FrameIdentifier frameID, PageIdentifier pageID, const WebCore::Cookie& cookie, ShouldRelaxThirdPartyCookieBlocking shouldRelaxThirdPartyCookieBlocking)
{
    if (m_hostsWithInMemoryStorage.contains<StringViewHashTranslator>(url.host()))
        inMemoryStorageSession().setCookieFromDOM(firstParty, sameSiteInfo, url, frameID, pageID, ApplyTrackingPrevention::No, cookie, shouldRelaxThirdPartyCookieBlocking);
}

void WebCookieCache::cookiesAdded(const String& host, const Vector<WebCore::Cookie>& cookies)
{
    if (!m_hostsWithInMemoryStorage.contains(host))
        return;

    for (auto& cookie : cookies)
        inMemoryStorageSession().setCookie(cookie);
}

void WebCookieCache::cookiesDeleted(const String& host, const Vector<WebCore::Cookie>& cookies)
{
    if (!m_hostsWithInMemoryStorage.contains(host))
        return;

    for (auto& cookie : cookies)
        inMemoryStorageSession().deleteCookie(cookie, [] { });
}

void WebCookieCache::allCookiesDeleted()
{
    clear();
}

void WebCookieCache::clear()
{
#if HAVE(COOKIE_CHANGE_LISTENER_API)
    for (auto& host : m_hostsWithInMemoryStorage)
        WebProcess::singleton().protectedCookieJar()->removeChangeListener(host, *this);
#endif
    m_hostsWithInMemoryStorage.clear();
    m_inMemoryStorageSession = nullptr;
}

void WebCookieCache::clearForHost(const String& host)
{
    String removedHost = m_hostsWithInMemoryStorage.take(host);
    if (removedHost.isNull())
        return;

    inMemoryStorageSession().deleteCookiesForHostnames(Vector<String> { removedHost }, [] { });
#if HAVE(COOKIE_CHANGE_LISTENER_API)
    WebProcess::singleton().protectedCookieJar()->removeChangeListener(removedHost, *this);
#endif
}

void WebCookieCache::pruneCacheIfNecessary()
{
    // We may want to raise this limit if we start using the cache for third-party iframes.
    static const unsigned maxCachedHosts = 5;

    while (m_hostsWithInMemoryStorage.size() >= maxCachedHosts)
        clearForHost(*m_hostsWithInMemoryStorage.random());
}

#if !PLATFORM(COCOA)
NetworkStorageSession& WebCookieCache::inMemoryStorageSession()
{
    ASSERT_NOT_IMPLEMENTED_YET();
    return *m_inMemoryStorageSession;
}

#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
void WebCookieCache::setOptInCookiePartitioningEnabled(bool)
{
    ASSERT_NOT_IMPLEMENTED_YET();
}
#endif
#endif

bool WebCookieCache::cacheMayBeOutOfSync() const
{
    return m_pendingCookieUpdateCounter.value() > 0;
}

} // namespace WebKit
