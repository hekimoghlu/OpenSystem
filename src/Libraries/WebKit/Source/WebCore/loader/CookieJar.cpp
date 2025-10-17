/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
#include "CookieJar.h"

#include "Cookie.h"
#include "CookieRequestHeaderFieldProxy.h"
#include "CookieStoreGetOptions.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "HTTPCookieAcceptPolicy.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "NetworkStorageSession.h"
#include "NetworkingContext.h"
#include "Page.h"
#include "PlatformStrategies.h"
#include "StorageSessionProvider.h"
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/SystemTracing.h>

namespace WebCore {

static ShouldRelaxThirdPartyCookieBlocking shouldRelaxThirdPartyCookieBlocking(const Document& document)
{
    if (RefPtr page = document.page())
        return page->shouldRelaxThirdPartyCookieBlocking();
    return ShouldRelaxThirdPartyCookieBlocking::No;
}

Ref<CookieJar> CookieJar::create(Ref<StorageSessionProvider>&& storageSessionProvider)
{
    return adoptRef(*new CookieJar(WTFMove(storageSessionProvider)));
}

IncludeSecureCookies CookieJar::shouldIncludeSecureCookies(const Document& document, const URL& url)
{
    return (url.protocolIs("https"_s) && !document.foundMixedContent().contains(SecurityContext::MixedContentType::Active)) ? IncludeSecureCookies::Yes : IncludeSecureCookies::No;
}

SameSiteInfo CookieJar::sameSiteInfo(const Document& document, IsForDOMCookieAccess isAccessForDOM)
{
    RefPtr frame = document.frame();
    if (frame && frame->loader().client().isRemoteWorkerFrameLoaderClient()) {
        RegistrableDomain domain(document.securityOrigin().data());
        return { domain.matches(document.firstPartyForCookies()), false, true };
    }

    if (RefPtr loader = document.loader())
        return SameSiteInfo::create(loader->request(), isAccessForDOM);
    return { };
}

CookieJar::CookieJar(Ref<StorageSessionProvider>&& storageSessionProvider)
    : m_storageSessionProvider(WTFMove(storageSessionProvider))
{
}

CookieJar::~CookieJar() = default;

String CookieJar::cookies(Document& document, const URL& url) const
{
    TraceScope scope(FetchCookiesStart, FetchCookiesEnd);

    auto includeSecureCookies = shouldIncludeSecureCookies(document, url);

    auto pageID = document.pageID();
    std::optional<FrameIdentifier> frameID;
    if (auto* frame = document.frame())
        frameID = frame->frameID();

    std::pair<String, bool> result;
    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession())
        result = session->cookiesForDOM(document.firstPartyForCookies(), sameSiteInfo(document, IsForDOMCookieAccess::Yes), url, frameID, pageID, includeSecureCookies, ApplyTrackingPrevention::Yes, shouldRelaxThirdPartyCookieBlocking(document));
    else
        ASSERT_NOT_REACHED();

    if (result.second)
        document.setSecureCookiesAccessed();

    return result.first;
}

CookieRequestHeaderFieldProxy CookieJar::cookieRequestHeaderFieldProxy(const Document& document, const URL& url)
{
    TraceScope scope(FetchCookiesStart, FetchCookiesEnd);

    auto pageID = document.pageID();
    std::optional<FrameIdentifier> frameID;
    if (auto* frame = document.frame())
        frameID = frame->frameID();

    return { document.firstPartyForCookies(), sameSiteInfo(document), url, frameID, pageID, shouldIncludeSecureCookies(document, url) };
}

void CookieJar::setCookies(Document& document, const URL& url, const String& cookieString)
{
    auto pageID = document.pageID();
    std::optional<FrameIdentifier> frameID;
    if (auto* frame = document.frame())
        frameID = frame->frameID();

    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession())
        session->setCookiesFromDOM(document.firstPartyForCookies(), sameSiteInfo(document, IsForDOMCookieAccess::Yes), url, frameID, pageID, ApplyTrackingPrevention::Yes, cookieString, shouldRelaxThirdPartyCookieBlocking(document));
    else
        ASSERT_NOT_REACHED();
}

bool CookieJar::cookiesEnabled(Document& document)
{
    auto cookieURL = document.cookieURL();
    if (cookieURL.isEmpty())
        return false;

    auto pageID = document.pageID();
    std::optional<FrameIdentifier> frameID;
    if (auto* frame = document.frame())
        frameID = frame->frameID();

    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession())
        return session->cookiesEnabled(document.firstPartyForCookies(), cookieURL, frameID, pageID, shouldRelaxThirdPartyCookieBlocking(document));

    ASSERT_NOT_REACHED();
    return false;
}

void CookieJar::remoteCookiesEnabled(const Document&, CompletionHandler<void(bool)>&& completionHandler) const
{
    completionHandler(false);
}

std::pair<String, SecureCookiesAccessed> CookieJar::cookieRequestHeaderFieldValue(const URL& firstParty, const SameSiteInfo& sameSiteInfo, const URL& url, std::optional<FrameIdentifier> frameID, std::optional<PageIdentifier> pageID, IncludeSecureCookies includeSecureCookies) const
{
    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession()) {
        std::pair<String, bool> result = session->cookieRequestHeaderFieldValue(firstParty, sameSiteInfo, url, frameID, pageID, includeSecureCookies, ApplyTrackingPrevention::Yes, ShouldRelaxThirdPartyCookieBlocking::No);
        return { result.first, result.second ? SecureCookiesAccessed::Yes : SecureCookiesAccessed::No };
    }

    ASSERT_NOT_REACHED();
    return { };
}

String CookieJar::cookieRequestHeaderFieldValue(Document& document, const URL& url) const
{
    auto pageID = document.pageID();
    std::optional<FrameIdentifier> frameID;
    if (auto* frame = document.frame())
        frameID = frame->frameID();

    auto result = cookieRequestHeaderFieldValue(document.firstPartyForCookies(), sameSiteInfo(document), url, frameID, pageID, shouldIncludeSecureCookies(document, url));
    if (result.second == SecureCookiesAccessed::Yes)
        document.setSecureCookiesAccessed();
    return result.first;
}

bool CookieJar::getRawCookies(Document& document, const URL& url, Vector<Cookie>& cookies) const
{
    auto pageID = document.pageID();
    std::optional<FrameIdentifier> frameID;
    if (auto* frame = document.frame())
        frameID = frame->frameID();

    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession())
        return session->getRawCookies(document.firstPartyForCookies(), sameSiteInfo(document), url, frameID, pageID, ApplyTrackingPrevention::Yes, shouldRelaxThirdPartyCookieBlocking(document), cookies);

    ASSERT_NOT_REACHED();
    return false;
}

void CookieJar::setRawCookie(const Document&, const Cookie& cookie, ShouldPartitionCookie)
{
    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession())
        session->setCookie(cookie);
    else
        ASSERT_NOT_REACHED();
}

void CookieJar::deleteCookie(const Document& document, const URL& url, const String& cookieName, CompletionHandler<void()>&& completionHandler)
{
    if (CheckedPtr session = protectedStorageSessionProvider()->storageSession())
        session->deleteCookie(document.firstPartyForCookies(), url, cookieName, WTFMove(completionHandler));
    else {
        ASSERT_NOT_REACHED();
        completionHandler();
    }
}

void CookieJar::getCookiesAsync(Document&, const URL&, const CookieStoreGetOptions&, CompletionHandler<void(std::optional<Vector<Cookie>>&&)>&& completionHandler) const
{
    completionHandler(std::nullopt);
}

void CookieJar::setCookieAsync(Document&, const URL&, const Cookie&, CompletionHandler<void(bool)>&& completionHandler) const
{
    completionHandler(false);
}

Ref<StorageSessionProvider> CookieJar::protectedStorageSessionProvider() const
{
    return m_storageSessionProvider;
}

#if HAVE(COOKIE_CHANGE_LISTENER_API)
void CookieJar::addChangeListener(const WebCore::Document&, const CookieChangeListener&)
{
}

void CookieJar::removeChangeListener(const String&, const CookieChangeListener&)
{
}
#endif

} // namespace WebCore
