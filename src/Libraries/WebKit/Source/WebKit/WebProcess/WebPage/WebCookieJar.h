/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#include "WebCookieCache.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/CookieChangeListener.h>
#include <WebCore/CookieJar.h>
#include <optional>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSHTTPCookieStorage;
#endif

namespace WebCore {
struct Cookie;
struct CookieStoreGetOptions;
enum class ShouldPartitionCookie : bool;
}

namespace WebKit {

class WebFrame;

class WebCookieJar final : public WebCore::CookieJar {
public:
    static Ref<WebCookieJar> create() { return adoptRef(*new WebCookieJar); }
    
    String cookies(WebCore::Document&, const URL&) const final;
    void setCookies(WebCore::Document&, const URL&, const String& cookieString) final;
    bool cookiesEnabled(WebCore::Document&) final;
    void remoteCookiesEnabled(const WebCore::Document&, CompletionHandler<void(bool)>&&) const final;
    std::pair<String, WebCore::SecureCookiesAccessed> cookieRequestHeaderFieldValue(const URL& firstParty, const WebCore::SameSiteInfo&, const URL&, std::optional<WebCore::FrameIdentifier>, std::optional<WebCore::PageIdentifier>, WebCore::IncludeSecureCookies) const final;
    bool getRawCookies(WebCore::Document&, const URL&, Vector<WebCore::Cookie>&) const final;
    void setRawCookie(const WebCore::Document&, const WebCore::Cookie&, WebCore::ShouldPartitionCookie) final;
    void deleteCookie(const WebCore::Document&, const URL&, const String& cookieName, CompletionHandler<void()>&&) final;

    void getCookiesAsync(WebCore::Document&, const URL&, const WebCore::CookieStoreGetOptions&, CompletionHandler<void(std::optional<Vector<WebCore::Cookie>>&&)>&&) const final;
    void setCookieAsync(WebCore::Document&, const URL&, const WebCore::Cookie&, CompletionHandler<void(bool)>&&) const final;

#if HAVE(COOKIE_CHANGE_LISTENER_API)
    void addChangeListenerWithAccess(const URL&, const URL& firstParty, WebCore::FrameIdentifier, WebCore::PageIdentifier, WebPageProxyIdentifier, const WebCore::CookieChangeListener&);
    void addChangeListener(const WebCore::Document&, const WebCore::CookieChangeListener&) final;
    void removeChangeListener(const String& host, const WebCore::CookieChangeListener&) final;
#endif

    void cookiesAdded(const String& host, Vector<WebCore::Cookie>&&);
    void cookiesDeleted(const String& host, Vector<WebCore::Cookie>&&);
    void allCookiesDeleted();

    void clearCache() final;

#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
    void setOptInCookiePartitioningEnabled(bool);
#endif

private:
    WebCookieJar();

    bool remoteCookiesEnabledSync(WebCore::Document&) const;
    void clearCacheForHost(const String&) final;
    bool isEligibleForCache(WebFrame&, const URL& firstPartyForCookies, const URL& resourceURL) const;
    String cookiesInPartitionedCookieStorage(const WebCore::Document&, const URL&, const WebCore::SameSiteInfo&) const;
    void setCookiesInPartitionedCookieStorage(const WebCore::Document&, const URL&, const WebCore::SameSiteInfo&, const String& cookieString);
#if PLATFORM(COCOA)
    NSHTTPCookieStorage* ensurePartitionedCookieStorage();
#endif

    mutable WebCookieCache m_cache;
    HashMap<String, WeakHashSet<WebCore::CookieChangeListener>> m_changeListeners;

#if PLATFORM(COCOA)
    RetainPtr<NSHTTPCookieStorage> m_partitionedStorageForDOMCookies;
#endif
};

} // namespace WebKit
